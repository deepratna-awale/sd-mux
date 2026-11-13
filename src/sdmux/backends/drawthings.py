"""
Draw Things (macOS / iOS) through its HTTP API server (Settings > API Server, HTTP).

    GET  /  or  /sdapi/v1/options    the app's current settings, one key per parameter
    POST /sdapi/v1/txt2img           {"images": [base64 PNG, ...]}
    POST /sdapi/v1/img2img           the same, from exactly one init image the size of width x height

Keys are the app's own (guidance_scale, sampler, model, loras [{file, weight}], strength, ...), with a few A1111
aliases. A key the app does not know fails the whole request with 422, so only keys the settings list are sent.
Omitted keys keep the app's values. There is no model or LoRA listing, no progress, no interrupt and no inpaint mask.
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from ..errors import Rejected, Unsupported
from ..names import canonical_sampler, canonical_scheduler, extract_loras, match_choice
from ..types import INPAINT, TXT2IMG, Model, Progress, Request, Result, infotext, load_image, to_b64
from .base import Backend

#: Draw Things sampler label -> (canonical sampler, canonical scheduler or None). "Trailing" timestep spacing is
#: closest to SGM Uniform; "AYS" is Align Your Steps.
SAMPLERS: dict[str, tuple[str, str | None]] = {
    "DPM++ 2M Karras": ("dpmpp_2m", "karras"),
    "Euler a": ("euler_ancestral", None),
    "DDIM": ("ddim", None),
    "PLMS": ("plms", None),
    "DPM++ SDE Karras": ("dpmpp_sde", "karras"),
    "UniPC": ("uni_pc", None),
    "LCM": ("lcm", None),
    "Euler A Substep": ("euler_ancestral", "substep"),
    "DPM++ SDE Substep": ("dpmpp_sde", "substep"),
    "TCD": ("tcd", None),
    "Euler A Trailing": ("euler_ancestral", "sgm_uniform"),
    "DPM++ SDE Trailing": ("dpmpp_sde", "sgm_uniform"),
    "DPM++ 2M AYS": ("dpmpp_2m", "align_your_steps"),
    "Euler A AYS": ("euler_ancestral", "align_your_steps"),
    "DPM++ SDE AYS": ("dpmpp_sde", "align_your_steps"),
    "DPM++ 2M Trailing": ("dpmpp_2m", "sgm_uniform"),
    "DDIM Trailing": ("ddim", "sgm_uniform"),
    "UniPC Trailing": ("uni_pc", "sgm_uniform"),
    "UniPC AYS": ("uni_pc", "align_your_steps"),
    "TCD Trailing": ("tcd", "sgm_uniform"),
}


def native_sampler(sampler: str | None, scheduler: str | None) -> str | None:
    """The Draw Things sampler for an A1111 sampler (and scheduler), or None when it has no counterpart."""
    if not sampler:
        return None
    if sampler in SAMPLERS and not scheduler:
        return sampler
    wanted, folded = canonical_sampler(sampler)
    schedule = canonical_scheduler(scheduler) if scheduler else folded
    candidates = [label for label, (s, _) in SAMPLERS.items() if s == wanted]
    if not candidates:
        return None
    for label in candidates:
        if SAMPLERS[label][1] == schedule:
            return label
    plain = [label for label in candidates if SAMPLERS[label][1] is None]
    return (plain or candidates)[0]


def _lora_file(name: str, files: list[str]) -> str:
    """A LoRA file the app has selected that matches name ("detail" -> "detail_lora_f16.ckpt"), else name unchanged."""
    found = match_choice(name, files)
    if found:
        return found
    for suffix in ("_lora_f16.ckpt", "_lora_q8p.ckpt", "_lora_f16", "_lora"):
        found = match_choice(name + suffix, files)
        if found:
            return found
    return name


class DrawThings(Backend):
    kind = "drawthings"
    label = "Draw Things"
    default_url = "http://127.0.0.1:7860"

    def settings(self) -> dict:
        """The app's current settings, as the API server reports them."""
        return self.http.get_json("/sdapi/v1/options", timeout=30)

    # Listings -------------------------------------------------------------
    def models(self) -> list[Model]:
        """Only the model selected in the app: Draw Things does not list its models over HTTP."""
        name = self.settings().get("model")
        return [Model(name=name)] if name else []

    def loras(self) -> list[Model]:
        """The LoRAs selected in the app."""
        return [Model(name=item["file"], kind="lora", raw=item) for item in self.settings().get("loras") or [] if item.get("file")]

    def samplers(self) -> list[str]:
        return list(SAMPLERS)

    def info(self) -> dict:
        return {"backend": self.kind, "url": self.url, "model": self.settings().get("model")}

    def current_model(self) -> str | None:
        return self.settings().get("model") or None

    def resolve_model(self, wanted: str | Model | None, models: list[Model] | None = None) -> Model:
        """The selected model, or wanted passed through: the app checks file names itself."""
        if isinstance(wanted, Model):
            return wanted
        if not wanted:
            return super().resolve_model(None, models)
        return Model(name=wanted)

    def resolve_loras(self, prompt: str, available: list[Model] | None = None) -> tuple[str, list[tuple[Model, float]]]:
        cleaned, wanted = extract_loras(prompt)
        if not wanted:
            return cleaned, []
        names = [m.name for m in (self.loras() if available is None else available)]
        return cleaned, [(Model(name=_lora_file(name, names), kind="lora"), weight) for name, weight in wanted]

    # Generation -----------------------------------------------------------
    def build_body(self, request: Request, settings: dict) -> dict:
        prompt, loras = self.resolve_loras(request.prompt, [Model(name=item["file"]) for item in settings.get("loras") or []
                                                            if item.get("file")])
        width, height = request.size(64)
        body: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": request.negative,
            "width": width,
            "height": height,
            "steps": request.steps,
            "guidance_scale": request.cfg,
            "seed": request.seed,
            "batch_size": request.batch_size,
        }
        if request.model:
            body["model"] = self.resolve_model(request.model).name
        if loras:
            body["loras"] = [{"file": m.name, "weight": weight} for m, weight in loras]
        if request.clip_skip:
            body["clip_skip"] = int(request.clip_skip)
        if request.sampler:
            sampler = native_sampler(request.sampler, request.scheduler)
            if sampler is None:
                raise Rejected(f"Draw Things has no sampler like {request.sampler!r}. It has: {', '.join(SAMPLERS)}",
                               backend=self.kind)
            body["sampler"] = sampler
        if request.mode != TXT2IMG:
            body["strength"] = request.denoise
        if request.hires is not None:
            # Draw Things' width/height is the final size; hires_fix_width/height is the first pass.
            if not request.hires.latent_method:
                raise Unsupported("Draw Things' hires fix upscales latents only; use upscaler=None.", backend=self.kind)
            body["width"], body["height"] = request.hires.size(width, height, 64)
            body.update(hires_fix=True, hires_fix_width=width, hires_fix_height=height, hires_fix_strength=request.hires.denoise)
        body = {k: v for k, v in body.items() if v is not None and (k in settings or not settings)}
        if request.image is not None and request.mode != TXT2IMG:
            image = load_image(request.image).convert("RGB")
            if image.size != (width, height):
                image = image.resize((width, height), Image.Resampling.LANCZOS)
            body["init_images"] = [to_b64(image)]
        body.update(request.extra or {})
        return body

    def _generate(self, request: Request) -> Result:
        if request.mode == INPAINT:
            raise Unsupported("Draw Things' HTTP API takes no mask; inpaint in the app or use img2img.", backend=self.kind)
        request.seed = request.resolved_seed() % (2**32)
        body = self.build_body(request, self.settings())
        route = "/sdapi/v1/txt2img" if request.mode == TXT2IMG else "/sdapi/v1/img2img"
        with self._job():
            response = self.http.request("POST", route, json=body, timeout=self.options.job_timeout)
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail") or response.text
            except (ValueError, AttributeError):
                detail = response.text
            raise Rejected(f"Draw Things rejected the request ({response.status_code}): {str(detail)[:1000]}",
                           backend=self.kind, details=response.text)
        encoded = self.http.json(response).get("images") or []
        if not encoded:
            raise Rejected("Draw Things returned no image.", backend=self.kind)
        model = body.get("model") or self.current_model()
        images = []
        for data in encoded:
            image = load_image(data)
            image.info["parameters"] = infotext(request, seed=request.seed, size=image.size, sampler=body.get("sampler"),
                                                model=model, version="Draw Things")
            images.append(image)
        return Result(images=images, seed=request.seed, seeds=[request.seed], parameters=images[0].info["parameters"],
                      backend=self.kind, raw={"body": {k: v for k, v in body.items() if k != "init_images"}})

    # Control --------------------------------------------------------------
    def progress(self) -> Progress:
        """Draw Things reports no progress over HTTP; the request simply returns when the image is done."""
        return Progress()

    def interrupt(self) -> None:
        raise Unsupported("Draw Things' HTTP API has no interrupt; stop the job in the app.", backend=self.kind)
