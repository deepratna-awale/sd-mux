# Verification notes

What was checked, against what, on 2026-11-11 and 2026-11-12.

## Items that were unverified in the brief

| Claim | Result | Source |
|---|---|---|
| A1111's `scheduler` request field needs 1.9 or newer | **Corrected: 1.10.0.** Added by commit `25cd53d7` ("scheduler selection in main UI", 2024-03-20); the first tags containing it are `v1.10.0-RC` / `v1.10.0`. Older servers read a scheduler glued to the sampler name (`"DPM++ 2M Karras"`), which sdmux sends when `/sdapi/v1/schedulers` is missing. | `stable-diffusion-webui` git history; `modules/sd_samplers.py` `get_sampler_and_scheduler` |
| A1111 inpaint fields are `mask`, `mask_blur`, `inpainting_fill`, `inpaint_full_res` | **Confirmed**, plus `inpaint_full_res_padding` and `inpainting_mask_invert` (0 = repaint white). Live inpaint left unmasked pixels unchanged. | `modules/api/models.py:122`, `modules/processing.py:1565-1570`; live A1111 1.10.1 |
| `override_settings_restore_afterwards` default reloads the previous model | **Confirmed.** Defaults to `True` (`modules/processing.py:175`); restoring `sd_model_checkpoint` after the job (`:853`) triggers its reload callback. sdmux sends `false`, so `model=` behaves like picking the model in the UI. | A1111 source |
| SwarmUI inpaint mask parameter is `maskimage` | **Confirmed.** The param is registered as "Mask Image" (white = changed), and param ids are the lower-cased name without spaces. Live inpaint works. | `src/Text2Image/T2IParamTypes.cs:515`, `CleanTypeName` at `:292`; live SwarmUI 0.9.8.4 |
| SwarmUI has no polling progress route; steps need `GenerateText2ImageWS` | **Confirmed.** `GetCurrentStatus` only counts waiting/live generations and loading models; step progress is the `gen_progress` message of `GenerateText2ImageWS`. sdmux reports activity only. | `src/WebAPI/BasicAPIFeatures.cs:561`, `src/WebAPI/T2IAPI.cs:38-75` |
| InvokeAI turns a mask image into the mask input with `image_mask_to_tensor` | **Corrected.** In 6.14 `create_gradient_mask` takes the mask as an *image* (`mask: ImageField`), so no tensor node is needed. Polarity: it repaints **black** and keeps white ("invert so 0 is unmasked"), so sdmux inverts the caller's white-is-repaint mask. `image_mask_to_tensor` exists but is not used. Live inpaint repainted only the masked area. | `invokeai/app/invocations/sd/create_gradient_mask.py:50,97`; `invokeai/app/invocations/mask.py:115-141`; live InvokeAI 6.14.4 |
| Civitai `by-hash` response shape | **Confirmed** with one call (a public SD 1.5 model's hash): `id`, `modelId`, `name`, `baseModel` ("SD 1.5"), `baseModelType`, `trainedWords`, `model {name, type, nsfw, poi}`, `files[]`, `air`, `downloadUrl`. An unknown hash answers 404. | live call 2026-11-11 |
| SDXL refiner has attn2 context width 1280 | **Confirmed.** `SDXLRefiner.unet_config` sets `context_dim: 1280` (and `adm_in_channels: 2560`). | ComfyUI `comfy/supported_models.py:167-172` |
| `input_blocks.0.0.weight` `shape[1]` is 9 for inpaint and 8 for instruct-pix2pix | **Confirmed for 8** (`SD15_instructpix2pix` / `SDXL_instructpix2pix` set `in_channels: 8`). ComfyUI treats any `in_channels > 4` as an inpaint model and does not name 9 itself; 9 is the standard SD inpainting UNet (4 latent + 4 masked-image latent + 1 mask), tested with a synthetic header. | `comfy/supported_models.py:508-529`, `comfy/supported_models_base.py:70` |
| ComfyUI function names in the brief | `detect_unet_config` is at line 44 and `unet_prefix_from_state_dict` at 1356 as stated. **`unet_config_from_diffusers_unet` does not exist**; the diffusers path is `model_config_from_diffusers_unet` at line 1577. | ComfyUI commit `29e28e2b` |
| Other model hosts with a hash lookup (added 2026-11-12) | civitai.red serves the same by-hash API as civitai.com (checked with a real call). CivArchive answers `GET /api/sha256/{hash}` with a `model` block (baseModel, platform, mirrors); an unknown hash still returns stray `files` but no `model`. modelindex.dev answers `GET /api/v1/hash/sha256:{hash}` with Hugging Face / ModelScope matches, 404 `HASH_UNKNOWN` otherwise. Hugging Face's `/api/models/{repo}` gives `cardData.base_model` and `diffusers:` tags. No public by-hash API was found for Hugging Face itself, Tensor.Art, SeaArt, Shakker or LiblibAI. | live calls 2026-11-12 |

## Live runs

Each backend ran `tests/live_check.py` with `hassakuSD15_v13` and the LoRA `2b_nierautomata-10` at 512x512, 8 steps:
detect, info, models, LoRAs, samplers, schedulers, `identify(model)`, `profile(model)`, txt2img, img2img, inpaint (top third
masked; the bottom pixel must be unchanged), and interrupt from a second thread during a 60-step job.

| Backend | Result |
|---|---|
| A1111 1.10.1 | all pass |
| Forge f2.0.1v1.10.1 | all pass (detected as `forge` after adding the `/sdapi/v1/sd-modules` probe) |
| reForge 739b2e1 | all pass (detected as `forge` from its launch flags) |
| SD.Next 6657d06 | all pass after the sigma/scheduler mapping and `/options` model switch fixes |
| ComfyUI 0.41.0 | all pass; also an SDXL txt2img with `waiIllustriousSDXL_v170` using its Illustrious profile |
| SwarmUI 0.9.8.4 (ComfyUI backend) | all pass after resolving the seed client-side and skipping mask blur without Swarm's nodes |
| InvokeAI 6.14.4 | all pass |

Bugs found live and fixed: A1111-family interrupts returned the partial image as a result (now `Cancelled`); Forge 2.x
was detected as plain A1111; SD.Next's `/schedulers` returns 500 and its `override_settings` model switch is ignored
when no model is loaded; SwarmUI's `donotsave` images carry no seed; SwarmUI mask blur needs Swarm's ComfyUI nodes;
a LoRA whose safetensors header length starts with byte `0x80` was mistaken for a pickle; legacy (non-zip) torch files
were read at the wrong pickle.

## identify() on the real files

Checkpoints (`--online`):

| File | kind / architecture / family / prediction | family decided by |
|---|---|---|
| waiIllustriousSDXL_v170.safetensors | checkpoint / sdxl / illustrious / epsilon | Civitai `baseModel=Illustrious` (offline: file name, low) |
| anyhentai_20.ckpt | checkpoint / sd15 / anime / epsilon | Civitai model name (pickle walked, never unpickled) |
| hassakuSD15_v13.safetensors | checkpoint / sd15 / anime / epsilon | Civitai model name |
| meinahentai_v4.safetensors | checkpoint / sd15 / anime / epsilon | merge recipe metadata |
| meinahentai_v5Final.safetensors | checkpoint / sd15 / anime / epsilon | Civitai model name |
| perfectWorld_v6Baked.safetensors | checkpoint / sd15 / - / epsilon | none (Civitai says SD 1.5) |

The two merge-recipe checkpoints with `model_ema.*` and `alphas_cumprod` keys identify cleanly.

LoRAs (364 files across A1111 `models/Lora` and ComfyUI `models/loras`, offline): every file identified as a LoRA.
185 sd15/anime, 149 sd15 with no family, 14 sdxl/pony, 8 sdxl/illustrious, 5 sdxl with no family, 2 flux, 1 sdxl/anime.
Family came from training metadata (`ss_sd_model_name`, e.g. Pony V6 and `animefull-final-pruned`) for 206 files,
from a sidecar for 1 and the file name for 1; 156 have no family evidence offline.

## Not verified / deferred

- ComfyUI split-file graphs (Flux, Flux.2, Chroma, Qwen-Image, SD3, Lumina 2): mock-only; no such models on the machine.
- ComfyUI websocket step progress: not observed live (activity was), not covered by a test.
- SwarmUI with its own ComfyUI nodes (mask blur and advanced features), and SwarmUI step progress over its websocket.
- InvokeAI SDXL graphs and LoRAs on SDXL: mock-only (only SD 1.5 models registered live).
- Logins (A1111 `--api-auth`, SwarmUI accounts, InvokeAI multi-user JWT) and https proxies: mock-only.
- GGUF headers: synthetic test only; no GGUF files on the machine.
- Pony vs Illustrious by weights: out of scope (weight-level fingerprinting is the next step).
- The A1111 install renders "Euler a" as noise on MPS with any client; not an sdmux issue.
