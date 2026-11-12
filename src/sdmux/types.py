"""Request and result types shared by every backend."""

from __future__ import annotations

import base64
import io
import random
from dataclasses import dataclass, field
from typing import Any, Union

from PIL import Image

ImageInput = Union[Image.Image, bytes, str, "os.PathLike[str]"]  # noqa: F821

TXT2IMG, IMG2IMG, INPAINT = "txt2img", "img2img", "inpaint"


def load_image(value) -> Image.Image:
    """A PIL image from a PIL image, raw bytes, a path, a base64 string or a data: URL."""
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, (bytes, bytearray)):
        image = Image.open(io.BytesIO(value))
        image.load()
        return image
    text = str(value)
    if text.startswith("data:"):
        text = text.split(",", 1)[1]
    elif len(text) < 4096:
        try:
            with open(text, "rb") as handle:
                return load_image(handle.read())
        except OSError:
            pass
    try:
        raw = base64.b64decode(text, validate=False)
        return load_image(raw)
    except (ValueError, OSError) as e:
        raise ValueError("Not an image, a path to one, or base64 image data.") from e


def to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    if image.mode not in ("RGB", "RGBA", "L"):
        image = image.convert("RGBA" if "A" in image.mode else "RGB")
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def to_b64(image: Image.Image) -> str:
    return base64.b64encode(to_png_bytes(image)).decode("ascii")


def random_seed() -> int:
    return random.randint(0, 2**32 - 1)


@dataclass
class Request:
    """
    One generation, in the unified vocabulary.

    mode is txt2img, img2img or inpaint. Names (model, sampler, scheduler, LoRAs in the
    prompt) are matched against what the backend lists. seed -1 means random; the seed
    actually used is reported in Result.seed. clip_skip uses A1111 numbering (1 = none,
    2 = skip the last layer). For inpaint, white mask pixels are repainted.

    extra holds backend-native fields merged into the native request as-is
    (A1111 payload keys, SwarmUI params, ...). Room for hires-fix, ControlNet and
    the like is left there and in future fields; they are not built yet.
    """

    prompt: str = ""
    negative: str = ""
    mode: str = TXT2IMG
    model: str | None = None
    width: int | None = None
    height: int | None = None
    steps: int = 20
    cfg: float = 7.0
    sampler: str | None = None
    scheduler: str | None = None
    seed: int = -1
    clip_skip: int | None = None
    batch_size: int = 1
    image: Any = None
    mask: Any = None
    denoise: float = 0.75
    mask_blur: int = 4
    inpaint_full_res: bool = False
    inpaint_padding: int = 32
    vae: str | None = None
    extra: dict = field(default_factory=dict)

    def resolved_seed(self) -> int:
        return self.seed if self.seed is not None and self.seed >= 0 else random_seed()


@dataclass
class Result:
    """What a generation returns. images are PIL images, each with A1111-style infotext in .info["parameters"]."""

    images: list[Image.Image]
    seed: int | None = None
    seeds: list[int] = field(default_factory=list)
    parameters: str = ""
    backend: str = ""
    raw: Any = None

    @property
    def image(self) -> Image.Image:
        return self.images[0]

    def save(self, path: str, index: int = 0) -> None:
        """Save one image as PNG with its infotext in the parameters chunk, like A1111 does."""
        from PIL import PngImagePlugin

        image = self.images[index]
        meta = PngImagePlugin.PngInfo()
        text = image.info.get("parameters") or self.parameters
        if text:
            meta.add_text("parameters", text)
        image.save(path, pnginfo=meta)


@dataclass
class Progress:
    """fraction is 0..1; step/steps when known; eta in seconds when the backend says."""

    fraction: float = 0.0
    step: int | None = None
    steps: int | None = None
    eta: float | None = None
    active: bool = False


@dataclass
class Model:
    """An entry of models() / loras(). path is set when the backend reports a file path."""

    name: str
    title: str | None = None
    path: str | None = None
    hash: str | None = None
    sha256: str | None = None
    kind: str = "checkpoint"
    architecture: str | None = None
    raw: Any = None

    def __str__(self) -> str:
        return self.name


def infotext(request: Request, *, seed: int | None, size: tuple[int, int], sampler: str | None = None,
             scheduler: str | None = None, model: str | None = None, version: str | None = None,
             loras: list[tuple[str, float]] | None = None) -> str:
    """A1111-style generation parameters text, so PNGs from any backend read back the same way."""
    from .names import a1111_sampler_label, a1111_scheduler_label, extract_loras, lora_tags

    prompt, parsed = extract_loras(request.prompt)
    tags = lora_tags(loras if loras is not None else parsed)
    fields = {
        "Steps": request.steps,
        "Sampler": a1111_sampler_label(sampler or request.sampler),
        "Schedule type": a1111_scheduler_label(scheduler or request.scheduler),
        "CFG scale": f"{request.cfg:g}",
        "Seed": seed,
        "Size": f"{size[0]}x{size[1]}",
        "Model": model or request.model,
        "Denoising strength": f"{request.denoise:g}" if request.mode != TXT2IMG else None,
        "Clip skip": request.clip_skip if request.clip_skip and request.clip_skip > 1 else None,
        "Version": version,
    }
    params = ", ".join(f"{k}: {v}" for k, v in fields.items() if v not in (None, ""))
    text = " ".join(part for part in (prompt, tags) if part)
    return f"{text}\nNegative prompt: {request.negative}\n{params}"
