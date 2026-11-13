# Changelog

All notable changes to sdmux are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `sdmux.aio`: an asyncio client with the same methods. Cancelling an awaited generation interrupts it on the server.
- `hires=` on `txt2img()` (a `Hires`, `True`, a scale or a dict) builds each backend's hires fix: latent, pixel or
  upscale-model upscaling followed by a second denoise pass.
- `upscalers()` and `upscale(image, upscaler, scale)` on A1111, Forge, SD.Next and ComfyUI.
- Draw Things (HTTP API server) and stable-diffusion.cpp's `sd-server` backends, detected by `sdmux.connect()`.
- `ComfyUI.cancel(prompt_id)` removes a pending prompt or interrupts a running one, leaving other clients' jobs alone.
- `SDMUX_NEGATIVE_TTL` sets how long a "not found" online lookup is trusted (default 7 days).
- The wheel ships `py.typed`, so type checkers use sdmux's annotations.
- `Request.size()`.

### Changed
- Pillow 9.1 or newer is required.
- CI fails on lint (ruff) and type (mypy) errors, and the command line has tests.

### Fixed
- A dropped request while looking up a model's profile no longer switches that model to the fixed fallback settings for
  the rest of the session.
- A ComfyUI job that passes `job_timeout` is removed from the queue or interrupted instead of left running.
- Online lookups that found nothing are asked again after the expiry window, so models published later are found.
  CivArchive 404s are cached like other misses.
- `interrupt()` is no longer lost when another call is waiting for the same client, and on ComfyUI it never stops
  another client's job.

## [0.1.0] - 2026-11-12

First release.

- `sdmux.connect()` with one API over AUTOMATIC1111, Forge / reForge, SD.Next, ComfyUI, SwarmUI and InvokeAI 5+:
  listings, txt2img, img2img, inpaint, progress, interrupt, typed errors, remote and authenticated servers.
- Defaults filled from the model's profile.
- ComfyUI node discovery and a validated graph builder.
- `sdmux.identify()` for safetensors, GGUF and torch checkpoints, with optional by-hash lookups on Civitai, CivArchive,
  modelindex.dev and Hugging Face.
- `sdmux` command line.

[Unreleased]: https://github.com/deepratna-awale/sdmux/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/deepratna-awale/sdmux/releases/tag/v0.1.0
