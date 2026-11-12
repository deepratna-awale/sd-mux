"""
Typed errors. Every failure a backend can report is one of these, so callers can
catch exactly what they care about:

    try:
        r = sd.txt2img("a cat")
    except sdmux.Unreachable:      # nothing answers at the URL
    except sdmux.AuthRefused:      # 401/403 or a failed login
    except sdmux.TimedOut:         # connected, but no answer in time
    except sdmux.NotReady:         # up, but missing a model, node, extension, ...
    except sdmux.Rejected:         # the backend refused or failed this request
    except sdmux.Cancelled:        # interrupt() stopped it

All of them derive from SdmuxError. Generation never returns None.
"""

from __future__ import annotations


class SdmuxError(RuntimeError):
    """Base class. code is a stable string id, handy for logs and HTTP APIs built on top."""

    code = "error"

    def __init__(self, message: str = "", *, backend: str | None = None, url: str | None = None, details=None):
        super().__init__(message)
        self.backend = backend
        self.url = url
        self.details = details

    def to_dict(self) -> dict:
        return {"code": self.code, "message": str(self), "backend": self.backend}


class Unreachable(SdmuxError, ConnectionError):
    """Nothing answered at the address (refused, DNS, TLS, connect timeout)."""

    code = "unreachable"


class TimedOut(SdmuxError, TimeoutError):
    """The server accepted the connection but did not answer, or a job did not finish, in time."""

    code = "timeout"


class AuthRefused(SdmuxError):
    """The server refused the credentials (HTTP 401/403 or a failed login)."""

    code = "auth_refused"


class NotReady(SdmuxError):
    """The server is up but lacks what the request needs (API flag, model, node, ...), or is not a supported backend."""

    code = "not_ready"


class Rejected(SdmuxError):
    """The server refused or failed this request (validation error, unknown model, exception during sampling)."""

    code = "rejected"


class Cancelled(SdmuxError):
    """The job was interrupted or cancelled."""

    code = "cancelled"


class Unsupported(NotReady):
    """The backend cannot do what was asked at all (for example inpainting through an API that has no mask input)."""

    code = "unsupported"


__all__ = ["SdmuxError", "Unreachable", "TimedOut", "AuthRefused", "NotReady", "Rejected", "Cancelled", "Unsupported"]
