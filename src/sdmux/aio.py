"""
asyncio front end for sdmux.

    import sdmux.aio

    async with await sdmux.aio.connect("http://127.0.0.1:8188") as sd:
        r = await sd.txt2img("1girl, smile", model="waiIllustrious")
        print((await sd.progress()).fraction)       # from another task while a generation runs
        await sd.interrupt()

Each call runs the matching method of the synchronous client in a worker thread, so the event loop never blocks and
several calls can be in flight at once (progress() and interrupt() while txt2img() waits). Cancelling a task that is
waiting on a generation, for example with asyncio.wait_for(), interrupts the job on the server before CancelledError
propagates. The synchronous client is available as .sync for anything not wrapped here.
"""

from __future__ import annotations

import asyncio
import functools
from typing import Any

from . import connect as _connect
from . import detect as _detect
from . import identify as _identify
from .backends.base import Backend
from .errors import SdmuxError
from .modelinfo import ModelInfo, Profile
from .probe import Detection
from .types import Model, Progress, Request, Result

_GENERATION = ("txt2img", "img2img", "inpaint", "generate", "upscale")


async def connect(url: str, kind: str | None = None, **kwargs) -> AsyncBackend:
    """sdmux.connect() without blocking the event loop."""
    return AsyncBackend(await asyncio.to_thread(_connect, url, kind, **kwargs))


async def detect(url: str, **option_overrides) -> Detection:
    return await asyncio.to_thread(_detect, url, **option_overrides)


async def identify(path, online: bool = False, **kwargs) -> ModelInfo:
    return await asyncio.to_thread(_identify, path, online, **kwargs)


class AsyncBackend:
    """An awaitable view of a Backend. Wraps a client made by sdmux.connect() or a backend class."""

    def __init__(self, backend: Backend) -> None:
        self.sync = backend

    @property
    def kind(self) -> str:
        return self.sync.kind

    @property
    def url(self) -> str:
        return self.sync.url

    def __repr__(self) -> str:
        return f"<sdmux.aio.AsyncBackend {self.sync.kind} {self.sync.url}>"

    async def _call(self, name: str, *args, **kwargs):
        method = getattr(self.sync, name)
        task = asyncio.ensure_future(asyncio.to_thread(method, *args, **kwargs))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if name in _GENERATION:
                try:
                    await asyncio.to_thread(self.sync.interrupt)
                except (SdmuxError, NotImplementedError):
                    pass
            raise

    # Listings -------------------------------------------------------------
    async def models(self) -> list[Model]:
        return await self._call("models")

    async def loras(self) -> list[Model]:
        return await self._call("loras")

    async def samplers(self) -> list[str]:
        return await self._call("samplers")

    async def schedulers(self) -> list[str]:
        return await self._call("schedulers")

    async def upscalers(self) -> list[str]:
        return await self._call("upscalers")

    async def info(self) -> dict:
        return await self._call("info")

    async def ready(self) -> dict:
        return await self._call("ready")

    # Generation -----------------------------------------------------------
    async def txt2img(self, prompt: str, negative: str = "", **kwargs) -> Result:
        return await self._call("txt2img", prompt, negative, **kwargs)

    async def img2img(self, image, prompt: str, negative: str = "", denoise: float = 0.75, **kwargs) -> Result:
        return await self._call("img2img", image, prompt, negative, denoise, **kwargs)

    async def inpaint(self, image, mask, prompt: str, negative: str = "", denoise: float = 0.75, **kwargs) -> Result:
        return await self._call("inpaint", image, mask, prompt, negative, denoise, **kwargs)

    async def generate(self, request: Request) -> Result:
        return await self._call("generate", request)

    async def upscale(self, image, upscaler: str | None = None, scale: float = 2.0) -> Result:
        return await self._call("upscale", image, upscaler, scale)

    # Control --------------------------------------------------------------
    async def progress(self) -> Progress:
        return await self._call("progress")

    async def interrupt(self) -> None:
        await self._call("interrupt")

    # Identification -------------------------------------------------------
    async def identify(self, model: str | Model, online: bool = False) -> ModelInfo:
        return await self._call("identify", model, online)

    async def profile(self, model: str | Model | None = None) -> Profile:
        return await self._call("profile", model)

    # Anything else --------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        """Other methods of the client (run_workflow, api, call, ...) as coroutines; plain attributes as they are."""
        attr = getattr(self.sync, name)
        if not callable(attr):
            return attr

        @functools.wraps(attr)
        async def call(*args, **kwargs):
            return await self._call(name, *args, **kwargs)

        return call

    async def close(self) -> None:
        await asyncio.to_thread(self.sync.close)

    async def __aenter__(self) -> AsyncBackend:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()
