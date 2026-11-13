"""
Work out which backend answers at a URL, with one cheap probe per API family:

    ComfyUI     GET  /system_stats             {"system": {...}, "devices": [...]}
    InvokeAI    GET  /api/v1/app/version       {"version": "5.x"}
    SwarmUI     POST /API/GetNewSession        {"session_id": ...} or a SwarmUI error_id
    A1111 API   GET  /sdapi/v1/samplers        a list; then A1111 vs Forge vs SD.Next from /version and /cmd-flags
"""

from __future__ import annotations

import re
from typing import Any, Callable
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from .errors import AuthRefused, NotReady, Rejected, TimedOut, Unreachable
from .transport import Options, Transport, redact_url

# Launch flags only Forge and reForge have, as /sdapi/v1/cmd-flags reports them.
FORGE_FLAGS = ("forge_ref_a1111_home", "cuda_malloc", "pin_shared_memory", "always_offload_from_vram", "unet_in_fp8_e4m3fn",
               "always_gpu", "always_high_vram", "always_normal_vram", "always_low_vram", "always_no_vram", "attention_pytorch",
               "disable_xformers", "forge_ref_comfy_home")

# RunPod's HTTP proxy: https://<pod id>-<port>.proxy.runpod.net. It answers on every port, exposed or not.
RUNPOD_PROXY = re.compile(r"^(?P<pod>[a-z0-9]+)-(?P<port>\d+)\.proxy\.runpod\.net$")


@dataclass
class Detection:
    kind: str | None
    evidence: str = ""
    version: str | None = None
    tried: list[str] = field(default_factory=list)


def _json(http: Transport, method: str, path: str, **kwargs):
    """The JSON a probe answered with; None when the route does not exist there. Auth and network errors raise."""
    response = http.request(method, path, timeout=10, **kwargs)
    if response.status_code >= 400:
        return None
    try:
        return response.json()
    except ValueError:
        return None


def _a1111_variant(http: Transport) -> tuple[str, str, str | None]:
    version = _json(http, "GET", "/sdapi/v1/version") or {}
    text = str(version).lower() if isinstance(version, dict) else ""
    label = None
    if isinstance(version, dict):
        label = version.get("version") or version.get("app") or None
    if "next" in str(version.get("app", "") if isinstance(version, dict) else "").lower() or "sdnext" in text:
        return "sdnext", "/sdapi/v1/version names SD.Next", label
    flags = _json(http, "GET", "/sdapi/v1/cmd-flags") or {}
    if isinstance(flags, dict):
        if any(flag in flags for flag in FORGE_FLAGS):
            return "forge", "Forge launch flags in /sdapi/v1/cmd-flags", label
        if "use_ipex" in flags and "backend" in flags:
            return "sdnext", "SD.Next launch flags in /sdapi/v1/cmd-flags", label
    # Forge 2.x (lllyasviel) parses its own flags elsewhere, so cmd-flags looks like A1111's;
    # it replaces /sdapi/v1/sd-vae with /sdapi/v1/sd-modules.
    if isinstance(_json(http, "GET", "/sdapi/v1/sd-modules"), list):
        return "forge", "/sdapi/v1/sd-modules (Forge 2.x)", label
    if isinstance(version, dict) and str(version.get("version", "")).startswith(("f", "classic")):
        return "forge", "Forge version string", label
    return "a1111", "/sdapi/v1 API", label


def detect(url: str, options: Options | None = None, **overrides) -> Detection:
    """
    Identify the backend at url.

    Raises:
        Unreachable: nothing answers at that address.
        TimedOut: it connected but did not answer.
        AuthRefused: something answered but refused the login, and no probe got through.
        NotReady: something answers, but it is not a supported backend (the message says what to check).
    """
    opts = Options.build(options, **overrides)
    opts.retries = min(opts.retries, 1)  # one retry covers a keep-alive connection the server dropped
    http = Transport(url, opts, "server")
    detection = Detection(None)
    refused: AuthRefused | None = None

    probes: tuple[tuple[str, str, str, dict, Callable[[Any], bool]], ...] = (
        ("comfyui", "GET", "/system_stats", {}, lambda r: isinstance(r, dict) and "system" in r and "devices" in r),
        ("invokeai", "GET", "/api/v1/app/version", {}, lambda r: isinstance(r, dict) and "version" in r),
        ("swarmui", "POST", "/API/GetNewSession", {"json": {}},
         lambda r: isinstance(r, dict) and ("session_id" in r or "error_id" in r)),
        ("a1111", "GET", "/sdapi/v1/samplers", {}, lambda r: isinstance(r, list)),
    )
    try:
        for kind, method, path, kwargs, matches in probes:
            detection.tried.append(f"{method} {path}")
            try:
                answer = _json(http, method, path, **kwargs)
            except AuthRefused as e:
                refused = refused or e
                continue
            except (Rejected, TimedOut):
                continue
            if not matches(answer):
                continue
            if kind == "a1111":
                kind, detection.evidence, detection.version = _a1111_variant(http)
            else:
                detection.evidence = f"{method} {path}"
                if kind == "comfyui":
                    detection.version = (answer.get("system") or {}).get("comfyui_version")
                elif kind == "invokeai":
                    detection.version = answer.get("version")
                elif kind == "swarmui":
                    detection.version = answer.get("version")
            detection.kind = kind
            return detection
    finally:
        http.close()

    if refused is not None:
        raise refused
    raise not_a_backend(url, detection)


def not_a_backend(url: str, detection: Detection) -> NotReady:
    """The error for an address that answers but is not a supported backend, with what to check for that host."""
    host = (urlsplit(url if "://" in url else f"http://{url}").hostname or "").lower()
    runpod = RUNPOD_PROXY.match(host)
    if runpod:
        port = runpod["port"]
        return NotReady(
            f"RunPod's proxy answers at {host}, but nothing behind it is a supported backend. Port {port} is probably "
            "not exposed on the pod, or the WebUI on it is not running or not listening on 0.0.0.0.",
            url=url,
        )
    return NotReady(
        f"Something answers at {redact_url(url)}, but not a backend sdmux supports (tried {', '.join(detection.tried)}). "
        "Check the port and any sub-path; A1111, Forge and reForge serve their API only when started with --api.",
        url=url,
    )
