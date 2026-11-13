"""
sdmux: one client for every local image generation backend, with checkpoint identification built in.

    import sdmux

    sd = sdmux.connect("http://127.0.0.1:8188")     # detects ComfyUI, A1111, Forge, SD.Next, SwarmUI, InvokeAI, ...
    r = sd.txt2img("1girl, smile <lora:detail:0.6>", model="waiIllustrious", steps=28, cfg=6,
                   sampler="Euler a", scheduler="Karras", seed=42, clip_skip=2)
    r.image.save("out.png")

    info = sdmux.identify("model.safetensors")       # reads only the header
    info.architecture, info.family, info.profile
"""

__version__ = "0.1.0"

from .comfygraph import Graph, GraphError, NodeSpec  # noqa: E402
from .errors import AuthRefused, Cancelled, NotReady, Rejected, SdmuxError, TimedOut, Unreachable, Unsupported  # noqa: E402
from .transport import Options  # noqa: E402
from .types import Hires, Model, Progress, Request, Result  # noqa: E402


def connect(url: str, kind: str | None = None, options: Options | None = None, defaults: str = "profile",
            **option_overrides):
    """
    Connect to the backend at url and return its client.

    kind skips detection ("a1111", "forge", "reforge", "sdnext", "comfyui", "swarmui", "invokeai", "drawthings", "sdcpp").
    Connection options: username, password, token, headers, connect_timeout, timeout, job_timeout,
    retries, verify_tls, poll_interval. Credentials in the URL (https://user:pass@host) work too.

    defaults: "profile" (default) fills settings a call leaves out from the model's profile; "fixed" uses
    sdmux's fixed size/steps/cfg and leaves sampler, scheduler and clip skip to the server.
    """
    from .backends import BACKENDS
    from .probe import detect

    if kind is None:
        kind = detect(url, options, **option_overrides).kind or ""
    try:
        cls = BACKENDS[kind.lower()]
    except KeyError:
        raise ValueError(f"Unknown backend kind {kind!r}. Known: {', '.join(sorted(BACKENDS))}") from None
    return cls(url, options, defaults=defaults, **option_overrides)


def detect(url: str, **option_overrides):
    """Which backend answers at url. Returns a Detection (kind, evidence, version, tried)."""
    from .probe import detect as _detect

    return _detect(url, **option_overrides)


def identify(path, online: bool = False, **kwargs):
    """Identify a model file from its header (and sidecars, metadata, name; Civitai when online=True)."""
    from .modelinfo import identify as _identify

    return _identify(path, online=online, **kwargs)


__all__ = [
    "connect", "detect", "identify", "Options", "Graph", "GraphError", "NodeSpec", "Request", "Result", "Progress", "Model", "Hires",
    "SdmuxError", "Unreachable", "TimedOut", "AuthRefused", "NotReady", "Rejected", "Cancelled", "Unsupported",
    "__version__",
]
