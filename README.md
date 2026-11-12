# sdmux

One Python client for every local image generation backend, with checkpoint identification built in.

- **One API** over AUTOMATIC1111, Forge / reForge, SD.Next, ComfyUI, SwarmUI and InvokeAI 5+.
- **One vocabulary in, native vocabulary out.** A1111 sampler and scheduler names (or each backend's own),
  `<lora:name:weight>` in the prompt everywhere, loose model and LoRA names.
- **Typed errors.** Generation raises; it never returns `None`.
- **Remote-friendly.** Localhost, a LAN address, or an https proxy with a sub-path, a login and extra headers.
- **`identify()`** tells a model file's kind, architecture, family, prediction type and variant from its header,
  sidecar files and metadata, and hands back a prompt profile with starting defaults.

Dependencies: `requests` and `pillow`. `websocket-client` (the `progress` extra) adds ComfyUI step progress.

```bash
pip install -e .              # or: pip install -e ".[progress]"
```

## Quickstart

```python
import sdmux

sd = sdmux.connect("http://127.0.0.1:8188")        # detects the backend
sd.kind                                            # "comfyui"
sd.models(); sd.loras(); sd.samplers(); sd.schedulers(); sd.info()

r = sd.txt2img("1girl, smile <lora:detail:0.6>", negative="lowres",
               model="waiIllustrious", width=1024, height=1024,
               steps=28, cfg=6, sampler="Euler a", scheduler="Karras",
               seed=42, clip_skip=2)
r.image, r.images, r.seed, r.parameters            # PIL images + A1111-style infotext
r.save("out.png")                                  # PNG with the infotext in its parameters chunk

sd.img2img(image, "prompt", denoise=0.5)
sd.inpaint(image, mask, "prompt", denoise=0.7)     # white = repaint
sd.progress(); sd.interrupt()                      # interrupt() from another thread -> Cancelled
```

Images can be PIL images, bytes, file paths or base64 strings. Every generation also accepts
`batch_size`, `vae`, `mask_blur`, `inpaint_full_res`, `inpaint_padding`, and `extra={...}` for
backend-native fields.

### Profiles: start from sensible defaults

```python
p = sd.profile("waiIllustrious")          # identify the listed model, then look up its profile
p.dialect                                 # "danbooru"
r = sd.txt2img(p.prompt("1girl, smile"), negative=p.negative_prompt("lowres"), model="waiIllustrious", **p.defaults)
```

Profiles are **starting points, not tuned truth**. Quality tags and negatives are seeded from
AutoExpress's per-family prompt sets.

### Remote servers

```python
sdmux.connect("https://user:pass@abc123-7860.proxy.runpod.net/sub/path")
sdmux.connect(url, token="...")                                   # bearer token (InvokeAI, SwarmUI's swarm_token)
sdmux.connect(url, headers={"CF-Access-Client-Id": "...", "CF-Access-Client-Secret": "..."})
sdmux.connect(url, timeout=60, job_timeout=3600, retries=3, verify_tls=False)
```

Credentials never appear in error messages (`***@host`), and extra headers are dropped if a redirect leaves the host.

### Errors

```python
try:
    sd.txt2img("a cat", model="nope")
except sdmux.Unreachable: ...   # nothing answers
except sdmux.TimedOut: ...      # connected, no answer in time (or job_timeout passed)
except sdmux.AuthRefused: ...   # 401/403 or a failed login
except sdmux.NotReady: ...      # up, but not a supported backend / missing what the request needs
except sdmux.Rejected: ...      # the backend refused or failed this request (unknown model, validation error, OOM)
except sdmux.Cancelled: ...     # interrupt() stopped it
```

`Unsupported` (a `NotReady`) means sdmux cannot build that request for that backend at all.
All errors derive from `sdmux.SdmuxError` and carry `.code`, `.backend`, `.details`.

### Escape hatches

| Backend | Native path |
|---|---|
| A1111 / Forge / SD.Next | `sd.api("txt2img", payload)` posts to any `/sdapi/v1` route; `extra=` merges into the payload |
| ComfyUI | `sd.run_workflow(api_format_dict)`; also `upload()`, `queue()`, `wait()`, `fetch()`, `node()`, `object_info()` |
| SwarmUI | `sd.call("Route", body)` with a live session; `extra=` merges into `GenerateText2Image` |
| InvokeAI | `sd.enqueue_graph(graph)`, `sd.wait(item_id)`, `sd.fetch(item)`, `sd.upload(image)` |

## Checkpoint identification

```python
info = sdmux.identify("model.safetensors")       # header only; never loads weights, never unpickles
info.kind            # checkpoint | diffusion_model | lora | vae | controlnet | text_encoder | embedding | unknown
info.architecture    # sd15 | sd2 | sdxl | sdxl-refiner | sd3 | flux | flux2 | chroma | qwen-image | wan | lumina2 | ...
info.family          # pony | illustrious | noobai | animagine | anime | furry | sdxl | ... (None when unknown)
info.prediction      # epsilon | v | flow | edm | None
info.variant         # inpaint | instruct | dev | schnell | fill | turbo | lightning | ...
info.evidence        # [Evidence(field, value, source, detail, confidence)]
info.conflicts       # where sources disagreed, and what was kept
info.profile         # dialect + defaults

sdmux.identify(path, online=True)                # adds Civitai's by-hash lookup (hash reused from sidecars/A1111 cache, else computed and cached)
sd.identify("hassaku")                           # a model the backend lists: reads the file when it is on this machine
```

Formats: `.safetensors`, `.gguf`, and torch `.ckpt` / `.pt` (zip or legacy). Pickle files are walked opcode by opcode
by a data-only stack machine; nothing in them is imported or executed.

Family sources, most trusted first: sidecar files (`.metadata.json` from ComfyUI LoRA Manager, `.civitai.info`,
`.civitai.json`, StabilityMatrix `.cm-info.json`, Civitai Browser+/Helper `.json`), Civitai (only with `online=True`),
`__metadata__` (`modelspec.*`, kohya `ss_*`, merge recipes), then the file name. Tensors decide kind and architecture.

```bash
sdmux detect http://127.0.0.1:7860
sdmux models http://127.0.0.1:8188 [--loras] [--json]
sdmux identify ~/models/checkpoints -r [--online] [--json] [-v]
sdmux txt2img http://127.0.0.1:7860 "a red apple" -o apple.png --model hassaku --steps 20
```

## Support matrix

**Live** = exercised against a real server on an Apple M4 (MPS) on 2026-11-12 with an SD 1.5 checkpoint and a LoRA
(ComfyUI also with an SDXL checkpoint). **Mock** = covered only by the offline fake-server tests.

| | A1111 1.10.1 | Forge 2.0.1 | reForge | SD.Next | ComfyUI 0.41 | SwarmUI 0.9.8.4 | InvokeAI 6.14.4 |
|---|---|---|---|---|---|---|---|
| Detection | Live | Live | Live (as `forge`) | Live | Live | Live | Live |
| Models / LoRAs / samplers / schedulers | Live | Live | Live | Live | Live | Live | Live (no separate schedulers) |
| txt2img (+ LoRA, clip skip, sampler/scheduler mapping) | Live | Live | Live | Live | Live | Live | Live |
| img2img | Live | Live | Live | Live | Live | Live | Live |
| inpaint (white = repaint) | Live | Live | Live | Live | Live | Live | Live |
| interrupt -> `Cancelled` | Live | Live | Live | Live | Live | Live | Live |
| progress | Live (steps) | Live (steps) | Live (steps) | Live (steps) | Live (active); websocket steps not verified | Live (active only) | Live (active only) |
| Login | Mock (basic) | Mock (basic) | Mock (basic) | Mock (basic) | Mock (basic/bearer) | Mock (`/API/Login`, token) | Mock (JWT) |
| `identify()` of listed models | Live (local paths) | Live | Live | Live | Live (local paths) | Live (name + reported arch) | Live (local paths) |

Model coverage for generation:

- **A1111 family, SwarmUI**: whatever the backend can load (sdmux passes the model name through).
- **ComfyUI**: single-file checkpoints (SD 1.x, SD 2.x, SDXL) are built with core nodes and live-verified.
  Split-file models in `models/diffusion_models` (Flux, Flux.2, Chroma, Qwen-Image, SD3, Lumina 2) get
  architecture-specific loaders chosen by `identify()`; those graphs are **mock-only** (no such models on the test machine).
  Anything else raises `Unsupported`; use `run_workflow()`.
- **InvokeAI**: SD 1.x, SD 2.x and SDXL main models. Flux, SD3 and others use other node families in InvokeAI
  and raise `Unsupported`; use `enqueue_graph()`.

Not built (room is left in `Request.extra`): hires-fix, refiner passes, ControlNet, IP-Adapter, upscalers, video.

## Backend notes

- **A1111 / Forge / reForge**: start with `--api`. A model chosen with `model=` stays loaded afterwards
  (`override_settings_restore_afterwards: false`); A1111's default would reload the previous model after every call.
  The `scheduler` field needs A1111 1.10+; on older versions the scheduler is appended to the sampler name.
  An interrupted job returns a partial image on these servers; sdmux raises `Cancelled` and keeps it in `details`.
- **SD.Next**: schedulers are diffusers classes picked by the sampler; the A1111 scheduler maps to SD.Next's
  "Sigma algorithm" (`schedulers_sigma`: karras, exponential, betas, ...). Models are switched through `/sdapi/v1/options`.
- **ComfyUI**: LoRAs become `LoraLoader` nodes, clip skip `CLIPSetLastLayer`, inpaint uses `SetLatentNoiseMask`
  and composites the original back outside the mask. `/view` downloads are retried when a proxy drops them.
- **SwarmUI**: LoRAs stay in the prompt. With `donotsave` the image carries no seed, so sdmux resolves the seed itself.
  Mask blur needs SwarmUI's own ComfyUI nodes; on a ComfyUI backend without them it is skipped.
- **InvokeAI**: the mask is inverted for `create_gradient_mask` (which repaints black), and the original pixels are
  composited back outside the mask.

## Development

```bash
pip install -e ".[test,progress]"
pytest                                            # offline: fake servers + synthetic headers, no GPU, no network
python tests/live_check.py http://127.0.0.1:7860 --model hassakuSD15_v13 --lora some_lora --out /tmp/live
```

See `docs/backends-setup.md` for the local UI installs used for live checks and `docs/verification.md` for what was
checked against which source.

## License

MIT
