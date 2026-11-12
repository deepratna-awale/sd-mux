# API coverage

Which native routes sdmux calls on each backend, and for which part of the unified API.
**Live** = exercised against the real server (see `verification.md`); **Mock** = only against the offline fake servers.
Anything not listed is reachable through the backend's escape hatch.

## Authentication

| Backend | What sdmux supports | How it is sent | Verified |
|---|---|---|---|
| A1111 / Forge / reForge / SD.Next | `--api-auth user:pass` (HTTP basic); a token for a reverse proxy | `username`/`password` or `https://user:pass@host` -> Basic; `token` -> `Authorization: Bearer` | Mock |
| ComfyUI | No auth of its own; basic or bearer auth from a proxy in front | Same as above, also on the progress websocket | Mock |
| SwarmUI | Accounts: `POST /API/Login {username, password}`; API tokens: `swarm_token` cookie (bound to the server's host) | `username`/`password`, or `token` | Mock |
| InvokeAI (multi-user) | `POST /api/v1/auth/login {email, password}` -> JWT, renewed once on a 401 | `username` = the account email, `password`; or a ready `token` | Mock |
| Any (login proxy, e.g. Cloudflare Access) | Extra headers on every request; dropped if a redirect leaves the host | `headers={...}` (dict, JSON, or `Name: value` lines) | Mock |

A refused login raises `AuthRefused`. Credentials are never shown in error messages (`***@host`) or in `repr()` of the options.

## AUTOMATIC1111, Forge, reForge, SD.Next (`/sdapi/v1`)

| Route | Used by | Notes | A1111 | Forge | reForge | SD.Next |
|---|---|---|---|---|---|---|
| `GET /sdapi/v1/samplers` | detection, `samplers()` | detection probe for the A1111 family | Live | Live | Live | Live |
| `GET /sdapi/v1/version` | detection, `info()` | names SD.Next; 404 on A1111 and Forge | Live | Live | Live | Live |
| `GET /sdapi/v1/cmd-flags` | detection | reForge's launch flags mark it as Forge | Live | Live | Live | Live |
| `GET /sdapi/v1/sd-modules` | detection | exists only on Forge 2.x | Live | Live | Live | - |
| `GET /sdapi/v1/sd-models` | `models()`, model matching, `identify()` (file paths) | | Live | Live | Live | Live |
| `GET /sdapi/v1/loras` | `loras()`, `<lora:>` matching | | Live | Live | Live | Live |
| `GET /sdapi/v1/schedulers` | `schedulers()` | 404 before A1111 1.10; 500 on SD.Next (sdmux lists its sigma options instead) | Live | Live | Live | Live |
| `GET /sdapi/v1/options` | `info()`, SD.Next model switch | | Live | Live | Live | Live |
| `POST /sdapi/v1/options` | SD.Next model switch | SD.Next ignores `override_settings` when no model is loaded | - | - | - | Live |
| `POST /sdapi/v1/txt2img` | `txt2img()` | `override_settings`: model, clip skip, VAE (SD.Next: `schedulers_sigma`, `clip_skip`) | Live | Live | Live | Live |
| `POST /sdapi/v1/img2img` | `img2img()`, `inpaint()` | inpaint: `mask`, `mask_blur`, `inpainting_fill=1`, `inpaint_full_res`, `inpaint_full_res_padding`, `inpainting_mask_invert=0` | Live | Live | Live | Live |
| `GET /sdapi/v1/progress?skip_current_image=true` | `progress()` | step, steps, eta | Live | Live | Live | Live |
| `POST /sdapi/v1/interrupt` | `interrupt()` | partial image -> `Cancelled` | Live | Live | Live | Live |
| any `/sdapi/v1/*` | `sd.api(path, payload, method)` | escape hatch | - | - | - | - |

## ComfyUI

| Route | Used by | Notes | Verified |
|---|---|---|---|
| `GET /system_stats` | detection, `info()` | version, python, devices | Live |
| `GET /object_info/{node}` | `models()`, `loras()`, `samplers()`, `schedulers()`, filling node defaults | cached per node; `refresh()` clears | Live |
| `GET /experiment/models`, `GET /experiment/models/{folder}` | file paths for `models()` / `loras()`, so `identify()` reads local files | | Live |
| `POST /upload/image` | `img2img()`, `inpaint()` | uploaded once per image content | Live |
| `POST /prompt` (with `client_id`) | every generation, `run_workflow()` | validation errors -> `Rejected`, one line per input | Live |
| `GET /history/{prompt_id}` | waiting for a job | execution errors -> `Rejected`, interrupts -> `Cancelled` | Live |
| `GET /view` | fetching output images | retried up to 4 times on a 404 or 5xx from a proxy | Live (retry: Mock) |
| `GET /queue` | `progress()` (activity) | | Live |
| `ws /ws?clientId=` | `progress()` steps | needs `websocket-client`; sends the same auth | Not verified |
| `POST /queue {"delete": [...]}`, `POST /interrupt` | `interrupt()` | drops the job if queued, interrupts it if running | Live |
| `GET /view_metadata/{folder}?filename=` | `sd.metadata(model)` | safetensors metadata read by the server (works remotely) | Mock |

Graph nodes: `CheckpointLoaderSimple`, `LoraLoader`, `CLIPSetLastLayer`, `CLIPTextEncode`, `EmptyLatentImage`, `LoadImage`,
`ImageScale`, `VAEEncode`, `LoadImageMask`, `SetLatentNoiseMask`, `RepeatLatentBatch`, `KSampler`, `VAEDecode`,
`ImageCompositeMasked`, `PreviewImage` (all Live). Split-file models add `UNETLoader`, `DualCLIPLoader` / `CLIPLoader` /
`TripleCLIPLoader`, `VAELoader`, `FluxGuidance`, `EmptySD3LatentImage`, `LoraLoaderModelOnly` (Mock).

## SwarmUI (`POST /API/<Route>`)

| Route | Used by | Notes | Verified |
|---|---|---|---|
| `GetNewSession` | detection, every call | renewed once on `invalid_session_id` | Live (renewal: Mock) |
| `Login` | accounts | | Mock |
| `ListModels` (`Stable-Diffusion`, `LoRA`) | `models()`, `loras()` | reports each model's architecture | Live |
| `ListT2IParams` | `samplers()`, `schedulers()` | `id///Display Name` values | Live |
| `GenerateText2Image` | `txt2img()`, `img2img()` (`initimage`, `initimagecreativity`), `inpaint()` (`maskimage`, `maskblur`) | LoRAs stay in the prompt; `donotsave`; seed resolved client-side | Live |
| `GetCurrentStatus` | `info()`, `ready()`, `progress()` (activity), `features()` | `supported_features` decides whether mask blur can be sent | Live |
| `InterruptAll` | `interrupt()` | this session only | Live |
| any route | `sd.call(route, body)` | escape hatch | - |

Not used: `GenerateText2ImageWS` (step progress), model downloads and editing.

## InvokeAI 5+

| Route | Used by | Notes | Verified |
|---|---|---|---|
| `GET /api/v1/app/version` | detection, `info()`, `ready()` | | Live |
| `POST /api/v1/auth/login` | multi-user login | | Mock |
| `GET /api/v2/models/?model_type=main\|lora` | `models()`, `loras()` | base -> architecture; file path for `identify()` | Live |
| `POST /api/v1/images/upload` | `img2img()`, `inpaint()` | the mask is inverted (InvokeAI repaints black) | Live |
| `POST /api/v1/queue/default/enqueue_batch` | every generation, `enqueue_graph()` | | Live |
| `GET /api/v1/queue/default/i/{id}` | waiting for a job | failed -> `Rejected`, canceled -> `Cancelled` | Live |
| `GET /api/v1/images/i/{name}/full` | fetching the output image | | Live |
| `GET /api/v1/queue/default/status` | `progress()` (activity) | | Live |
| `PUT /api/v1/queue/default/i/{id}/cancel` | `interrupt()` | | Live |

Graph nodes: `main_model_loader`, `clip_skip`, `lora_selector` + `collect` + `lora_collection_loader`, `compel`, `noise`,
`denoise_latents`, `i2l`, `create_gradient_mask`, `l2i` (Live, SD 1.5); `sdxl_model_loader`, `sdxl_compel_prompt`,
`sdxl_lora_collection_loader` (Mock).
