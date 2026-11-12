"""The interface every backend implements, and the parts they share."""

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING

from ..errors import NotReady, Rejected
from ..errors import SdmuxError
from ..names import canonical_scheduler, extract_loras, lora_stem, match_choice
from ..transport import Options, Transport
from ..types import IMG2IMG, INPAINT, TXT2IMG, Model, Progress, Request, Result, load_image

if TYPE_CHECKING:
    from ..modelinfo import ModelInfo, Profile


class Backend:
    """
    A connection to one image generation server.

    Listings: models(), loras(), samplers(), schedulers(), info().
    Generation: txt2img(), img2img(), inpaint(), or generate(Request). They raise on failure.
    Control: progress(), interrupt().
    Identification: identify(model), profile(model).

    Every backend also exposes its native request path (see each subclass) so nothing the
    unified surface lacks blocks you.
    """

    kind = ""
    label = ""
    default_url = "http://127.0.0.1:7860"

    #: Fixed values for settings neither the caller nor the model profile provides.
    FALLBACK = {"width": 512, "height": 512, "steps": 20, "cfg": 7.0}

    def __init__(self, url: str | None = None, options: Options | None = None, defaults: str = "profile",
                 **option_overrides) -> None:
        """
        defaults: "profile" (settings a call leaves out come from the model's profile: size, steps, cfg,
        sampler, scheduler, clip skip) or "fixed" (sdmux's FALLBACK size, steps and cfg; sampler, scheduler
        and clip skip are left to the server). Whatever is still missing after the profile gets FALLBACK.
        """
        if defaults not in ("profile", "fixed"):
            raise ValueError("defaults must be 'profile' or 'fixed'.")
        self.defaults = defaults
        self._profiles: dict = {}
        self.options = Options.build(options, **option_overrides)
        self.http = Transport(url or self.default_url, self.options, self.label)
        self.options = self.http.options
        self._lock = threading.Lock()
        self._cancel = threading.Event()

    # Basics ---------------------------------------------------------------
    @property
    def url(self) -> str:
        return self.http.base_url

    def __repr__(self) -> str:
        return f"<sdmux.{type(self).__name__} {self.kind} {self.url}>"

    def close(self) -> None:
        self.http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # Listings -------------------------------------------------------------
    def models(self) -> list[Model]:
        raise NotImplementedError

    def loras(self) -> list[Model]:
        return []

    def samplers(self) -> list[str]:
        return []

    def schedulers(self) -> list[str]:
        return []

    def info(self) -> dict:
        return {"backend": self.kind, "url": self.url}

    def ready(self) -> dict:
        """
        A slim readiness check: {"ready": bool, "problems": [str], "info": {...}}.

        Raises Unreachable / AuthRefused when the server cannot be talked to at all.
        """
        problems = []
        info = self.info()
        if not self.models():
            problems.append(f"{self.label} lists no models.")
        return {"ready": not problems, "problems": problems, "info": info}

    # Generation -----------------------------------------------------------
    def txt2img(self, prompt: str, negative: str = "", **kwargs) -> Result:
        return self.generate(Request(prompt=prompt, negative=negative, mode=TXT2IMG, **kwargs))

    def img2img(self, image, prompt: str, negative: str = "", denoise: float = 0.75, **kwargs) -> Result:
        return self.generate(Request(prompt=prompt, negative=negative, mode=IMG2IMG, image=image, denoise=denoise, **kwargs))

    def inpaint(self, image, mask, prompt: str, negative: str = "", denoise: float = 0.75, **kwargs) -> Result:
        """White mask pixels are repainted, black ones kept."""
        return self.generate(
            Request(prompt=prompt, negative=negative, mode=INPAINT, image=image, mask=mask, denoise=denoise, **kwargs)
        )

    def generate(self, request: Request) -> Result:
        request = self._prepare(request)
        filled = self._fill_defaults(request)
        self._cancel.clear()
        result = self._generate(request)
        result.defaults = filled
        return result

    def current_model(self) -> str | None:
        """The model a request without model= runs on. Most backends use the first listed one."""
        return None

    def _profile_for(self, model):
        key = model.name if isinstance(model, Model) else (model or self.current_model() or "")
        if key not in self._profiles:
            try:
                resolved = self.resolve_model(model or self.current_model())
                self._profiles[key] = (resolved.name, self.identify(resolved).profile)
            except (SdmuxError, OSError, ValueError):
                self._profiles[key] = (None, None)
        return self._profiles[key]

    def _fill_defaults(self, request: Request) -> dict:
        """Fill settings the caller left as None; returns {field: (value, source)}."""
        filled: dict = {}

        def put(name, value, source):
            if getattr(request, name) is None and value is not None:
                setattr(request, name, value)
                filled[name] = (value, source)

        if self.defaults == "profile":
            name, profile = self._profile_for(request.model)
            if profile is not None:
                source = f"profile:{name}"
                if request.mode == TXT2IMG:
                    put("width", profile.width, source)
                    put("height", profile.height, source)
                put("steps", profile.steps, source)
                put("cfg", profile.cfg, source)
                if request.sampler is None and request.scheduler is None:
                    put("sampler", profile.sampler, source)
                    scheduler = profile.scheduler if profile.scheduler and profile.scheduler.lower() != "automatic" else None
                    put("scheduler", scheduler, source)
                put("clip_skip", profile.clip_skip, source)
        # Size, steps and cfg are always sent, so whatever is still missing gets sdmux's fixed values.
        for name, value in self.FALLBACK.items():
            put(name, value, "fallback")
        return filled

    def _generate(self, request: Request) -> Result:
        raise NotImplementedError

    def _prepare(self, request: Request) -> Request:
        """Validate the request and load its images once, so backends receive PIL images."""
        if request.mode not in (TXT2IMG, IMG2IMG, INPAINT):
            raise ValueError(f"Unknown mode {request.mode!r}")
        if request.mode in (IMG2IMG, INPAINT):
            if request.image is None:
                raise ValueError(f"{request.mode} needs an image.")
            request.image = load_image(request.image)
            if request.width is None:
                request.width = request.image.width
            if request.height is None:
                request.height = request.image.height
        if request.mode == INPAINT:
            if request.mask is None:
                raise ValueError("inpaint needs a mask (white = repaint).")
            request.mask = load_image(request.mask)
        if request.scheduler and canonical_scheduler(request.scheduler) == "automatic":
            request.scheduler = None  # the backend's own default
        if request.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        return request

    # Control --------------------------------------------------------------
    def progress(self) -> Progress:
        return Progress()

    def interrupt(self) -> None:
        raise NotImplementedError

    # Shared helpers -------------------------------------------------------
    def resolve_model(self, wanted: "str | Model | None", models: list[Model] | None = None) -> Model:
        """The listed model matching wanted (loosely), or the first one when wanted is None. A Model passes through."""
        if isinstance(wanted, Model):
            return wanted
        models = self.models() if models is None else models
        if not models:
            raise NotReady(f"{self.label} lists no models.", backend=self.kind)
        if not wanted:
            return models[0]
        for attr in ("name", "title", "path"):
            names = [getattr(m, attr) for m in models if getattr(m, attr)]
            found = match_choice(wanted, names)
            if found:
                return next(m for m in models if getattr(m, attr) == found)
        raise Rejected(
            f"{self.label} has no model matching {wanted!r}. It lists: "
            + ", ".join(m.name for m in models[:20]) + (" ..." if len(models) > 20 else ""),
            backend=self.kind,
        )

    def resolve_loras(self, prompt: str, available: list[Model] | None = None) -> tuple[str, list[tuple[Model, float]]]:
        """Strip <lora:...> tags from prompt and match each against the backend's LoRAs. Unknown LoRAs raise Rejected."""
        cleaned, wanted = extract_loras(prompt)
        if not wanted:
            return cleaned, []
        available = self.loras() if available is None else available
        found = []
        for name, weight in wanted:
            match = None
            for attr in ("name", "title", "path"):
                names = [getattr(m, attr) for m in available if getattr(m, attr)]
                hit = match_choice(name, names)
                if hit:
                    match = next(m for m in available if getattr(m, attr) == hit)
                    break
            if match is None:
                raise Rejected(f"{self.label} has no LoRA matching {name!r}.", backend=self.kind)
            found.append((match, weight))
        return cleaned, found

    @staticmethod
    def lora_name(model: Model) -> str:
        return lora_stem(model.name)

    # Identification -------------------------------------------------------
    def local_path(self, model: Model) -> str | None:
        """The model's file on this machine, when the backend reports a path that exists here."""
        if model.path and os.path.isfile(model.path):
            return model.path
        return None

    def identify(self, model: str | Model, online: bool = False) -> "ModelInfo":
        """
        identify() for a listed model. Reads the file when it is reachable from this machine,
        otherwise works from the name and whatever the backend reports about it.
        """
        from ..modelinfo import identify, identify_name

        if not isinstance(model, Model):
            model = self.resolve_model(model)
        path = self.local_path(model)
        if path:
            return identify(path, online=online)
        return identify_name(model.name, architecture=model.architecture, kind=model.kind, sha256=model.sha256, online=online)

    def profile(self, model: str | Model | None = None) -> "Profile":
        """Prompt dialect and starting defaults for a model this backend lists."""
        if model is None or not isinstance(model, Model):
            model = self.resolve_model(model)
        return self.identify(model).profile
