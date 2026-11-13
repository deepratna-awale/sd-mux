import asyncio

import pytest
from PIL import Image

import sdmux
import sdmux.aio

from .fakes import a1111_server, comfyui_server


def test_async_client_mirrors_sync():
    async def main(url):
        assert (await sdmux.aio.detect(url)).kind == "a1111"
        async with await sdmux.aio.connect(url) as sd:
            assert sd.kind == "a1111" and isinstance(sd.sync, sdmux.backends.A1111)
            assert "hassakuSD15_v13" in [m.name for m in await sd.models()]
            assert await sd.samplers() and await sd.upscalers()
            r = await sd.txt2img("a cat", seed=5, steps=4)
            assert r.seed == 5 and r.image
            r = await sd.img2img(Image.new("RGB", (64, 64)), "a dog", denoise=0.3)
            assert r.image
            assert (await sd.progress()).active
            assert await sd.api("samplers", method="GET")  # anything else, through __getattr__
            assert sd.restore_settings is False  # plain attributes pass through

    with a1111_server() as s:
        asyncio.run(main(s.url))


def test_async_progress_interrupt_and_cancel_while_generating():
    async def main(url):
        sd = await sdmux.aio.connect(url, poll_interval=0.05)
        job = asyncio.create_task(sd.txt2img("x"))
        await asyncio.sleep(0.3)
        assert not job.done()
        await sd.progress()  # answers while the generation waits
        await sd.interrupt()
        with pytest.raises(sdmux.Cancelled):
            await job

        # Cancelling the awaiting task interrupts the job on the server.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(sd.txt2img("y"), 0.3)
        return sd

    with comfyui_server() as s:
        s.state["mode"] = "hang"
        asyncio.run(main(s.url))
        interrupts = [r.json for r in s.requests if (r.method, r.path) == ("POST", "/interrupt")]
        assert interrupts == [{"prompt_id": "p1"}, {"prompt_id": "p2"}]
