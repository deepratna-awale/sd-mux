"""
SwarmUI, through its /API/<Route> calls (every call is a POST carrying a session_id).

LoRAs stay as <lora:name:weight> in the prompt, which SwarmUI parses itself. When SwarmUI
requires accounts, pass token= (sent as the swarm_token cookie) or username/password
(logged in through /API/Login).

Native escape hatch: call(route, body) posts to any /API route with a live session;
generate() merges Request.extra into the GenerateText2Image body, so every T2I param
SwarmUI lists (ListT2IParams) is reachable.
"""

from __future__ import annotations

import io
import json

from PIL import Image

from ..errors import AuthRefused, Cancelled, Rejected
from ..names import canonical_scheduler, extract_loras, lora_tags, pick_native, sampler_id, split_option
from ..types import IMG2IMG, INPAINT, Model, Progress, Request, Result, infotext, load_image, to_b64
from .base import Backend

# SwarmUI model class ids -> sdmux architecture names.
ARCHITECTURES = {
    "stable-diffusion-v1": "sd15",
    "stable-diffusion-v1-inpainting": "sd15",
    "stable-diffusion-v2": "sd2",
    "stable-diffusion-v2-768-v": "sd2",
    "stable-diffusion-v2-inpainting": "sd2",
    "stable-diffusion-xl-v1-base": "sdxl",
    "stable-diffusion-xl-v1-refiner": "sdxl-refiner",
    "stable-diffusion-xl-v1-edit": "sdxl",
    "stable-diffusion-v3-medium": "sd3",
    "stable-diffusion-v3.5-large": "sd3",
    "stable-diffusion-v3.5-medium": "sd3",
    "flux.1-dev": "flux",
    "flux.1-schnell": "flux",
    "flux.2-dev": "flux2",
    "chroma": "chroma",
    "qwen-image": "qwen-image",
}


def architecture_of(swarm_arch: str | None) -> str | None:
    if not swarm_arch:
        return None
    arch = swarm_arch.split("/", 1)[0].lower()
    if arch in ARCHITECTURES:
        return ARCHITECTURES[arch]
    for prefix, name in (("stable-diffusion-xl", "sdxl"), ("stable-diffusion-v1", "sd15"), ("stable-diffusion-v2", "sd2"),
                         ("stable-diffusion-v3", "sd3"), ("flux.2", "flux2"), ("flux.1", "flux"), ("qwen-image", "qwen-image"),
                         ("wan-2", "wan"), ("hidream", "hidream"), ("lumina", "lumina2"), ("z-image", "lumina2")):
        if arch.startswith(prefix):
            return name
    return None


class SwarmUI(Backend):
    kind = "swarmui"
    label = "SwarmUI"
    default_url = "http://127.0.0.1:7801"

    def __init__(self, url=None, options=None, **kw) -> None:
        super().__init__(url, options, **kw)
        # SwarmUI reads its token from a cookie and does its own login: no HTTP auth goes out.
        self.http.session.auth = None
        self.http.session.headers.pop("Authorization", None)
        if self.options.token:
            # Bound to the server's host so a redirect elsewhere never carries it
            # (the cookie jar matches dotless hosts such as localhost as "<host>.local").
            from urllib.parse import urlsplit

            host = urlsplit(self.url).hostname or ""
            self.http.session.cookies.set("swarm_token", self.options.token, domain=host if "." in host else f"{host}.local")
        self._session_id: str | None = None
        self._logged_in = False
        self._params: dict[str, dict] | None = None
        self._server: dict = {}
        self._features: list[str] | None = None

    # Session --------------------------------------------------------------
    def _log_in(self) -> None:
        if not self.options.username or self._logged_in or self.options.token:
            return
        result = self.http.post_json("/API/Login", {"username": self.options.username, "password": self.options.password or ""})
        if result.get("error_id") or result.get("error"):
            raise AuthRefused(f"SwarmUI login failed: {result.get('error') or result.get('error_id')}", backend=self.kind)
        self._logged_in = True

    def _new_session(self) -> str:
        self._log_in()
        result = self.http.post_json("/API/GetNewSession", {})
        if "session_id" not in result:
            raise AuthRefused(f"SwarmUI did not open a session: {result.get('error') or result.get('error_id') or result}",
                              backend=self.kind)
        self._server = {k: result.get(k) for k in ("version", "server_id", "user_id", "permissions") if k in result}
        self._session_id = result["session_id"]
        return self._session_id

    def call(self, route: str, body: dict | None = None, timeout: float | None = None) -> dict:
        """POST /API/<route> with a session, renewing the session once if SwarmUI dropped it. Errors raise Rejected."""
        for attempt in range(2):
            session_id = self._session_id or self._new_session()
            result = self.http.post_json(f"/API/{route}", {"session_id": session_id, **(body or {})}, timeout=timeout)
            if result.get("error_id") == "invalid_session_id" and attempt == 0:
                self._session_id = None
                continue
            if "error" in result or "error_id" in result:
                message = str(result.get("error") or result.get("error_id"))
                if self._cancel.is_set() or "interrupt" in message.lower() or "cancel" in message.lower():
                    raise Cancelled(f"SwarmUI: {message}", backend=self.kind)
                raise Rejected(f"SwarmUI: {message}", backend=self.kind, details=result)
            return result
        raise AuthRefused("SwarmUI keeps rejecting the session.", backend=self.kind)

    # Listings -------------------------------------------------------------
    def params(self, refresh: bool = False) -> dict[str, dict]:
        """ListT2IParams keyed by param id."""
        if self._params is None or refresh:
            self._params = {p["id"]: p for p in self.call("ListT2IParams").get("list", [])}
        return self._params

    def param_values(self, param_id: str) -> list[str]:
        return [split_option(v)[0] for v in (self.params().get(param_id, {}).get("values") or [])]

    def _list(self, subtype: str, kind: str) -> list[Model]:
        result = self.call("ListModels", {"path": "", "depth": 20, "subtype": subtype})
        return [
            Model(name=f["name"], title=f.get("title"), kind=kind, architecture=architecture_of(f.get("architecture")), raw=f)
            for f in result.get("files", [])
        ]

    def models(self) -> list[Model]:
        return self._list("Stable-Diffusion", "checkpoint")

    def loras(self) -> list[Model]:
        return self._list("LoRA", "lora")

    def samplers(self) -> list[str]:
        return self.param_values("sampler")

    def schedulers(self) -> list[str]:
        return self.param_values("scheduler")

    def features(self) -> list[str]:
        """SwarmUI's supported_features (from its backends): 'comfyui', 'variation_seed' (Swarm's core nodes), ..."""
        if self._features is None:
            self._features = list(self.call("GetCurrentStatus").get("supported_features") or [])
        return self._features

    def info(self) -> dict:
        if not self._server:
            self._new_session()
        status: dict = {}
        try:
            status = self.call("GetCurrentStatus").get("backend_status") or {}
        except Rejected:
            pass
        return {"backend": self.kind, "url": self.url, **self._server, "backend_status": status}

    def ready(self) -> dict:
        problems = []
        info = self.info()
        status = (info.get("backend_status") or {}).get("status")
        if status in ("errored", "all_disabled", "empty", "disabled"):
            problems.append(f"SwarmUI backends are {status}: {(info.get('backend_status') or {}).get('message', '')}".strip())
        if not self.models():
            problems.append("SwarmUI lists no Stable-Diffusion models.")
        return {"ready": not problems, "problems": problems, "info": info}

    # Generation -----------------------------------------------------------
    def build_body(self, request: Request) -> dict:
        prompt, loras = extract_loras(request.prompt)
        if loras:
            _, matched = self.resolve_loras(request.prompt)
            # SwarmUI wants the LoRA's listed name (folder kept, extension dropped).
            prompt = " ".join(p for p in (prompt, lora_tags([(_strip_ext(m.name), w) for m, w in matched])) if p)
        model = self.resolve_model(request.model)
        body = {
            "images": request.batch_size,
            "prompt": prompt,
            "negativeprompt": request.negative,
            "model": model.name,
            "steps": request.steps,
            "cfgscale": request.cfg,
            # Resolved here: with donotsave SwarmUI returns images without the metadata that would name the seed.
            "seed": request.resolved_seed() % (2**31),
            "width": request.width,
            "height": request.height,
            "aspectratio": "Custom",
            "donotsave": True,
        }
        if request.sampler:
            body["sampler"] = pick_native(request.sampler, self.samplers(), sampler_id) or request.sampler
        scheduler = request.scheduler or _folded(request.sampler)
        if scheduler:
            body["scheduler"] = pick_native(scheduler, self.schedulers(), canonical_scheduler) or scheduler
        if request.clip_skip and request.clip_skip > 1:
            body["clipstopatlayer"] = -int(request.clip_skip)
        if request.vae:
            body["vae"] = request.vae
        if request.mode in (IMG2IMG, INPAINT):
            body["initimage"] = to_b64(request.image.convert("RGB"))
            body["initimagecreativity"] = request.denoise
        if request.mode == INPAINT:
            body["maskimage"] = to_b64(request.mask.convert("L"))
            # Mask blur runs through SwarmUI's own ComfyUI nodes (SwarmMaskBlur). A ComfyUI backend without
            # them (no "variation_seed" feature) rejects the whole workflow, so blur is skipped there.
            if request.mask_blur and "variation_seed" in self.features():
                body["maskblur"] = request.mask_blur
            if request.inpaint_full_res:
                body["maskshrinkgrow"] = request.inpaint_padding
        body.update(request.extra or {})
        return body

    def _image(self, ref: str) -> Image.Image:
        if ref.startswith("data:"):
            return load_image(ref)
        response = self.http.check(self.http.request("GET", "/" + ref.lstrip("/"), timeout=120), "image download")
        image = Image.open(io.BytesIO(response.content))
        image.load()
        return image

    def _generate(self, request: Request) -> Result:
        body = self.build_body(request)
        with self._job():
            result = self.call("GenerateText2Image", body, timeout=self.options.job_timeout)
            cancelled = self._cancel.is_set()
        refs = result.get("images") or []
        if not refs:
            if cancelled:
                raise Cancelled("Interrupted.", backend=self.kind)
            raise Rejected("SwarmUI returned no image.", backend=self.kind, details=result)
        images, seeds = [], []
        for ref in refs:
            image = self._image(ref["image"] if isinstance(ref, dict) else ref)
            params = _swarm_params(image)
            seed = params.get("seed", body["seed"])
            seeds.append(seed)
            text = infotext(request, seed=seed, size=image.size, sampler=params.get("sampler") or body.get("sampler"),
                            scheduler=params.get("scheduler") or body.get("scheduler"), model=body["model"],
                            version=f"SwarmUI {self._server.get('version') or ''}".strip())
            image.info["swarm_metadata"] = image.info.get("parameters")
            image.info["parameters"] = text
            images.append(image)
        return Result(images=images, seed=seeds[0], seeds=seeds, parameters=images[0].info["parameters"],
                      backend=self.kind, raw={"response": result, "body": {k: v for k, v in body.items() if k not in ("initimage", "maskimage")}})

    # Control --------------------------------------------------------------
    def progress(self) -> Progress:
        """SwarmUI's HTTP API reports whether generations are running, not their steps (that is only on its websocket)."""
        status = self.call("GetCurrentStatus").get("status") or {}
        active = bool(status.get("live_gens") or status.get("waiting_gens") or status.get("loading_models"))
        return Progress(active=active)

    def interrupt(self) -> None:
        self._cancel.set()
        self.call("InterruptAll", {"other_sessions": False}, timeout=10)


def _strip_ext(name: str) -> str:
    for ext in (".safetensors", ".ckpt", ".pt", ".bin"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return name


def _folded(sampler: str | None) -> str | None:
    from ..names import canonical_sampler

    return canonical_sampler(sampler)[1] if sampler else None


def _swarm_params(image: Image.Image) -> dict:
    """SwarmUI stores its parameters as JSON in the PNG 'parameters' chunk."""
    raw = image.info.get("parameters")
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return data.get("sui_image_params") or {}
