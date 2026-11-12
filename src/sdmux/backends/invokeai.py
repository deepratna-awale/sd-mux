"""
InvokeAI 5 and later, through its session queue API.

Each request becomes a graph of core nodes: model loader, optional clip skip and LoRA
collection, compel prompts, noise, denoise_latents, l2i; img2img adds i2l, inpaint adds
create_gradient_mask. SD 1.x, SD 2.x and SDXL main models are covered. Flux, SD3 and the
other architectures use separate node families in InvokeAI and are not built by sdmux;
enqueue_graph() takes any graph you build yourself.

Multi-user servers take token=, or username (the account email) and password, which are
exchanged for a token through /api/v1/auth/login and again when it expires.
"""

from __future__ import annotations

import io
import time

from PIL import Image, ImageOps

from ..errors import AuthRefused, Cancelled, NotReady, Rejected, TimedOut, Unsupported
from ..names import canonical_sampler, canonical_scheduler
from ..types import IMG2IMG, INPAINT, TXT2IMG, Model, Progress, Request, Result, infotext, to_png_bytes
from .base import Backend

QUEUE = "/api/v1/queue/default"
SUPPORTED_BASES = ("sd-1", "sd-2", "sdxl")
BASE_ARCHITECTURE = {"sd-1": "sd15", "sd-2": "sd2", "sdxl": "sdxl", "sdxl-refiner": "sdxl-refiner", "sd-3": "sd3",
                     "flux": "flux", "flux2": "flux2", "cogview4": "cogview4", "z-image": "lumina2", "qwen-image": "qwen-image"}

# InvokeAI folds the scheduler into the sampler name; "_k" variants use Karras sigmas.
SCHEDULERS = [
    "ddim", "ddpm", "deis", "deis_k", "lms", "lms_k", "pndm", "heun", "heun_k", "euler", "euler_k", "euler_a",
    "kdpm_2", "kdpm_2_k", "kdpm_2_a", "kdpm_2_a_k", "dpmpp_2s", "dpmpp_2s_k", "dpmpp_2m", "dpmpp_2m_k",
    "dpmpp_2m_sde", "dpmpp_2m_sde_k", "dpmpp_3m", "dpmpp_3m_k", "dpmpp_sde", "dpmpp_sde_k", "er_sde", "unipc", "unipc_k",
    "lcm", "tcd",
]
# canonical sampler id -> InvokeAI scheduler
SAMPLER_MAP = {
    "euler": "euler",
    "euler_ancestral": "euler_a",
    "heun": "heun",
    "lms": "lms",
    "dpm_2": "kdpm_2",
    "dpm_2_ancestral": "kdpm_2_a",
    "dpmpp_2s_ancestral": "dpmpp_2s",
    "dpmpp_sde": "dpmpp_sde",
    "dpmpp_2m": "dpmpp_2m",
    "dpmpp_2m_sde": "dpmpp_2m_sde",
    "dpmpp_3m_sde": "dpmpp_3m",
    "ddim": "ddim",
    "ddpm": "ddpm",
    "uni_pc": "unipc",
    "lcm": "lcm",
    "deis": "deis",
    "tcd": "tcd",
    "plms": "pndm",
}


def scheduler_name(sampler: str | None, scheduler: str | None) -> str:
    """An A1111 or native sampler (+ scheduler) as InvokeAI's single scheduler name."""
    if sampler in SCHEDULERS:
        return sampler
    if not sampler:
        return "euler"
    cid, folded = canonical_sampler(sampler)
    if cid not in SAMPLER_MAP:
        raise Rejected(f"InvokeAI has no sampler matching {sampler!r}. It has: {', '.join(SCHEDULERS)}", backend="invokeai")
    name = SAMPLER_MAP[cid]
    if canonical_scheduler(scheduler or folded) == "karras" and f"{name}_k" in SCHEDULERS:
        name = f"{name}_k"
    return name


def identifier(model: dict) -> dict:
    return {key: model[key] for key in ("key", "hash", "name", "base", "type") if key in model}


class InvokeAI(Backend):
    kind = "invokeai"
    label = "InvokeAI"
    default_url = "http://127.0.0.1:9090"

    def __init__(self, url=None, options=None, **kw) -> None:
        super().__init__(url, options, **kw)
        # InvokeAI has no basic auth: credentials are traded for a token.
        self.http.session.auth = None
        self._jwt: str | None = None
        self._item_id: int | None = None
        self._models_raw: dict[str, list[dict]] = {}
        original = self.http.request

        def request(method, path, timeout=None, **kwargs):
            has_login = bool(self.options.username) and not self.options.token
            if has_login and not self._jwt:
                self._log_in(original)
            try:
                return original(method, path, timeout, **kwargs)
            except AuthRefused:
                if not has_login:
                    raise
                self._log_in(original)  # the token may have expired
                return original(method, path, timeout, **kwargs)

        self.http.request = request  # type: ignore[method-assign]

    def _log_in(self, send) -> None:
        self.http.session.headers.pop("Authorization", None)
        response = send("POST", "/api/v1/auth/login", json={"email": self.options.username, "password": self.options.password or ""})
        if response.status_code != 200:
            raise AuthRefused(f"InvokeAI login failed ({response.status_code}): {response.text[:300]}", backend=self.kind)
        self._jwt = response.json()["token"]
        self.http.session.headers["Authorization"] = f"Bearer {self._jwt}"

    # Listings -------------------------------------------------------------
    def _raw_models(self, model_type: str) -> list[dict]:
        models = self.http.get_json("/api/v2/models/", params={"model_type": model_type}).get("models", [])
        self._models_raw[model_type] = models
        return models

    @staticmethod
    def _model(m: dict, kind: str) -> Model:
        return Model(name=m["name"], title=m.get("description") or None, path=m.get("path"), hash=m.get("hash"),
                     sha256=(m.get("hash") or "").split(":", 1)[-1] if str(m.get("hash", "")).startswith("sha256:") else None,
                     kind=kind, architecture=BASE_ARCHITECTURE.get(m.get("base")), raw=m)

    def models(self) -> list[Model]:
        """Main models. Every base is listed; generation covers sd-1, sd-2 and sdxl."""
        return [self._model(m, "checkpoint") for m in self._raw_models("main")]

    def loras(self) -> list[Model]:
        return [self._model(m, "lora") for m in self._raw_models("lora")]

    def samplers(self) -> list[str]:
        return list(SCHEDULERS)

    def info(self) -> dict:
        version = self.http.get_json("/api/v1/app/version", timeout=10)
        return {"backend": self.kind, "url": self.url, "version": version.get("version")}

    def ready(self) -> dict:
        problems = []
        info = self.info()
        major = str(info.get("version") or "0").split(".")[0]
        if not major.isdigit() or int(major) < 5:
            problems.append(f"InvokeAI {info.get('version')} is too old; sdmux needs 5 or newer.")
        if not [m for m in self.models() if m.raw.get("base") in SUPPORTED_BASES]:
            problems.append("InvokeAI has no SD 1.x, SD 2.x or SDXL main model.")
        return {"ready": not problems, "problems": problems, "info": info}

    def local_path(self, model: Model) -> str | None:
        """InvokeAI reports paths relative to its models folder for installed models; absolute for in-place ones."""
        return super().local_path(model)

    # Graph ----------------------------------------------------------------
    def build_graph(self, request: Request, image_name: str | None = None, mask_name: str | None = None) -> dict:
        model = self.resolve_model(request.model)
        raw = model.raw
        if raw.get("base") not in SUPPORTED_BASES:
            raise Unsupported(f"sdmux builds InvokeAI graphs for SD 1.x, SD 2.x and SDXL; {model.name} is {raw.get('base')}. "
                              "Use enqueue_graph() with your own graph.", backend=self.kind)
        sdxl = raw["base"] == "sdxl"
        prompt, loras = self.resolve_loras(request.prompt)
        nodes: dict[str, dict] = {}
        edges: list[dict] = []

        def node(node_id: str, node_type: str, **fields) -> str:
            nodes[node_id] = {"id": node_id, "type": node_type, "is_intermediate": True, "use_cache": False, **fields}
            return node_id

        def edge(source: str, source_field: str, destination: str, destination_field: str) -> None:
            edges.append({"source": {"node_id": source, "field": source_field},
                          "destination": {"node_id": destination, "field": destination_field}})

        loader = node("model_loader", "sdxl_model_loader" if sdxl else "main_model_loader", model=identifier(raw))
        unet_source, clip_source, clip2_source = loader, loader, loader
        if request.clip_skip and request.clip_skip > 1:
            if sdxl:
                pass  # SDXL in InvokeAI has no clip_skip on the dual encoder path
            else:
                clip_source = node("clip_skip", "clip_skip", skipped_layers=int(request.clip_skip) - 1)
                edge(loader, "clip", clip_source, "clip")

        selected = []
        for lora, weight in loras:
            if lora.raw.get("base") != raw["base"]:
                raise Rejected(f"LoRA {lora.name} is for {lora.raw.get('base')}, the model is {raw['base']}.", backend=self.kind)
            selected.append(node(f"lora_{len(selected)}", "lora_selector", lora=identifier(lora.raw), weight=weight))
        if selected:
            collect = node("lora_collect", "collect")
            for item in selected:
                edge(item, "lora", collect, "item")
            lora_loader = node("lora_loader", "sdxl_lora_collection_loader" if sdxl else "lora_collection_loader")
            edge(collect, "collection", lora_loader, "loras")
            edge(unet_source, "unet", lora_loader, "unet")
            edge(clip_source, "clip", lora_loader, "clip")
            if sdxl:
                edge(loader, "clip2", lora_loader, "clip2")
                clip2_source = lora_loader
            unet_source, clip_source = lora_loader, lora_loader

        width, height = int(request.width) // 8 * 8, int(request.height) // 8 * 8
        for prompt_id, text in (("positive", prompt), ("negative", request.negative)):
            if sdxl:
                node(prompt_id, "sdxl_compel_prompt", prompt=text, style=text, original_width=width, original_height=height,
                     crop_top=0, crop_left=0, target_width=width, target_height=height)
                edge(clip_source, "clip", prompt_id, "clip")
                edge(clip2_source, "clip2", prompt_id, "clip2")
            else:
                node(prompt_id, "compel", prompt=text)
                edge(clip_source, "clip", prompt_id, "clip")

        noise = node("noise", "noise", seed=request.seed, width=width, height=height, use_cpu=True)
        denoising_start = 0.0 if request.mode == TXT2IMG else round(1 - min(max(float(request.denoise), 0.0), 1.0), 4)
        denoise = node("denoise", "denoise_latents", steps=request.steps, cfg_scale=request.cfg,
                       scheduler=scheduler_name(request.sampler, request.scheduler),
                       denoising_start=denoising_start, denoising_end=1.0)
        edge(unet_source, "unet", denoise, "unet")
        edge("positive", "conditioning", denoise, "positive_conditioning")
        edge("negative", "conditioning", denoise, "negative_conditioning")
        edge(noise, "noise", denoise, "noise")

        if request.mode in (IMG2IMG, INPAINT):
            i2l = node("i2l", "i2l", image={"image_name": image_name})
            edge(loader, "vae", i2l, "vae")
            edge(i2l, "latents", denoise, "latents")
        if request.mode == INPAINT:
            mask = node("gradient_mask", "create_gradient_mask", mask={"image_name": mask_name},
                        edge_radius=max(int(request.mask_blur), 0) * 2, coherence_mode="Gaussian Blur", minimum_denoise=0.0,
                        image={"image_name": image_name})
            edge(loader, "vae", mask, "vae")
            edge(unet_source, "unet", mask, "unet")
            edge(mask, "denoise_mask", denoise, "denoise_mask")

        output = node("output", "l2i")
        nodes[output]["is_intermediate"] = False
        edge(denoise, "latents", output, "latents")
        edge(loader, "vae", output, "vae")
        return {"id": "sdmux", "nodes": nodes, "edges": edges}

    # Running --------------------------------------------------------------
    def upload(self, image: Image.Image, category: str = "user") -> str:
        response = self.http.request(
            "POST", "/api/v1/images/upload",
            params={"image_category": category, "is_intermediate": "true"},
            files={"file": ("sdmux.png", to_png_bytes(image), "image/png")},
            timeout=120,
        )
        return self.http.json(self.http.check(response, "image upload"))["image_name"]

    def enqueue_graph(self, graph: dict, runs: int = 1) -> int:
        """Enqueue any graph; returns the first queue item id."""
        response = self.http.request("POST", f"{QUEUE}/enqueue_batch",
                                     json={"batch": {"graph": graph, "runs": runs}, "prepend": False})
        if response.status_code not in (200, 201):
            raise Rejected(f"InvokeAI rejected the graph: {response.text[:2000]}", backend=self.kind, details=response.text)
        body = self.http.json(response)
        item_ids = body.get("item_ids") or []
        if not item_ids:
            raise Rejected("InvokeAI did not return a queue item; sdmux needs InvokeAI 5 or newer.", backend=self.kind, details=body)
        return item_ids[0]

    def wait(self, item_id: int, timeout: float | None = None) -> dict:
        deadline = time.monotonic() + (timeout or self.options.job_timeout)
        while time.monotonic() < deadline:
            item = self.http.get_json(f"{QUEUE}/i/{item_id}")
            status = item.get("status")
            if status == "completed":
                return item
            if status == "canceled":
                raise Cancelled("Cancelled.", backend=self.kind)
            if status == "failed":
                raise Rejected(f"InvokeAI failed: {item.get('error_type') or ''} {item.get('error_message') or ''}".strip(),
                               backend=self.kind, details=item.get("error_traceback"))
            time.sleep(self.options.poll_interval)
        raise TimedOut(f"InvokeAI job {item_id} did not finish within {timeout or self.options.job_timeout:.0f}s.", backend=self.kind)

    def fetch(self, item: dict, node: str = "output") -> Image.Image:
        session = item.get("session") or {}
        prepared = (session.get("source_prepared_mapping") or {}).get(node) or []
        results = session.get("results") or {}
        output = next((results[p] for p in prepared if p in results), None)
        if not output or "image" not in output:
            raise Rejected("InvokeAI finished without an output image.", backend=self.kind)
        name = output["image"]["image_name"]
        response = self.http.check(self.http.request("GET", f"/api/v1/images/i/{name}/full", timeout=120), "image download")
        image = Image.open(io.BytesIO(response.content))
        image.load()
        return image

    def _generate(self, request: Request) -> Result:
        request.seed = request.resolved_seed() % (2**31)  # InvokeAI noise seeds are 32-bit signed
        width, height = int(request.width) // 8 * 8, int(request.height) // 8 * 8
        image_name = mask_name = None
        work = None
        if request.image is not None:
            work = request.image.convert("RGB")
            if work.size != (width, height):
                work = work.resize((width, height), Image.LANCZOS)
            image_name = self.upload(work)
        if request.mask is not None:
            # create_gradient_mask repaints black and keeps white (checked in InvokeAI's source), so invert ours.
            mask = request.mask.convert("L").resize((width, height), Image.NEAREST)
            mask_name = self.upload(ImageOps.invert(mask), "mask")
        images, seeds = [], []
        for i in range(request.batch_size):
            seed = (request.seed + i) % (2**31)
            graph = self.build_graph(Request(**{**request.__dict__, "seed": seed}), image_name, mask_name)
            with self._lock:
                self._item_id = self.enqueue_graph(graph)
                try:
                    item = self.wait(self._item_id)
                finally:
                    self._item_id = None
            image = self.fetch(item)
            if request.mode == INPAINT and work is not None:
                # Keep pixels outside the mask exactly as they were.
                keep = request.mask.convert("L").resize(image.size, Image.NEAREST)
                image = Image.composite(image.convert("RGB"), work.resize(image.size), keep)
            text = infotext(request, seed=seed, size=image.size, sampler=request.sampler, scheduler=request.scheduler,
                            model=self.resolve_model(request.model).name, version="InvokeAI")
            image.info["parameters"] = text
            images.append(image)
            seeds.append(seed)
            if self._cancel.is_set():
                raise Cancelled("Interrupted.", backend=self.kind)
        return Result(images=images, seed=seeds[0], seeds=seeds, parameters=images[0].info["parameters"], backend=self.kind)

    # Control --------------------------------------------------------------
    def progress(self) -> Progress:
        status = self.http.get_json(f"{QUEUE}/status", timeout=10)
        queue = status.get("queue") or {}
        return Progress(active=bool(queue.get("in_progress") or queue.get("pending")))

    def interrupt(self) -> None:
        self._cancel.set()
        item_id = self._item_id
        if item_id is not None:
            self.http.check(self.http.request("PUT", f"{QUEUE}/i/{item_id}/cancel", timeout=10), "cancel")
