# Local backend installs

The UIs used to live-verify sdmux on the Apple M4 / 16 GB / macOS test machine, set up on 2026-11-11 and 12.
Everything lives on `/Volumes/980Pro`. Run one UI at a time and stop it before starting the next.

## Shared environment

`/Volumes/980Pro/Workspace/AI/.caches/env.sh` points every cache at the 980Pro. Source it before
installing or starting any UI:

```bash
export C=/Volumes/980Pro/Workspace/AI/.caches
export PIP_CACHE_DIR=$C/pip UV_CACHE_DIR=$C/uv HF_HOME=$C/hf TORCH_HOME=$C/torch TMPDIR=$C/tmp
export UV_PYTHON_INSTALL_DIR=$C/uv-python XDG_CACHE_HOME=$C NUGET_PACKAGES=$C/nuget DOTNET_CLI_HOME=$C/dotnet-home
export DOTNET_ROOT=/Volumes/980Pro/Workspace/AI/dotnet DOTNET_CLI_TELEMETRY_OPTOUT=1 PATH=/Volumes/980Pro/Workspace/AI/dotnet:$PATH
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

No checkpoints were downloaded. Every UI reads the existing models. **SwarmUI and SD.Next write next to the model
files they scan** (SwarmUI rewrites safetensors headers to add `modelspec` metadata and thumbnails, SD.Next drops
`.thumb.jpg` files), so they get their own model folders made of per-file symlinks, never a symlink to a whole folder.
If new checkpoints or LoRAs appear in A1111's folders, re-run the link loops below.

| UI | Folder | Version | Python | Port | Models |
|---|---|---|---|---|---|
| AUTOMATIC1111 (existing) | `AI/stable-diffusion-webui` | v1.10.1 | its `venv` (3.10) | 7860 | its own |
| ComfyUI (existing) | `AI/ComfyUI` | 0.41.0, commit 29e28e2b | miniconda 3.12 via comfy-cli | 8188 | its own + A1111's via `extra_model_paths.yaml` |
| Forge | `AI/forge` | f2.0.1v1.10.1, commit dfdcbab | `venv` (Homebrew 3.10) | 7863 | A1111's, via `--ckpt-dir` / `--lora-dir` |
| reForge | `AI/reforge` | commit 739b2e1 | `venv` (Homebrew 3.10) | 7861 | A1111's, via `--ckpt-dir` / `--lora-dir` |
| SD.Next | `AI/sdnext` | commit 6657d06 (2026-10-29) | `venv` (uv, 3.12) | 7862 | `sdnext/models/{Stable-diffusion,Lora}`: per-file symlinks |
| SwarmUI | `AI/SwarmUI` | 0.9.8.4, commit f86858d | .NET 8 SDK in `AI/dotnet` | 7801 | `SwarmUI/Models/{Stable-Diffusion,Lora}`: per-file symlinks; generates through the existing ComfyUI |
| InvokeAI | `AI/InvokeAI` (root + `.venv`) | 6.14.4 | `.venv` (uv, 3.12) | 9090 | registered in place (no copies) |
| InvokeAI source | `AI/InvokeAI-src` | commit 5b776e3 | - | - | read for node definitions only |

## AUTOMATIC1111 (existing, unchanged)

```bash
cd /Volumes/980Pro/Workspace/AI/stable-diffusion-webui
venv/bin/python launch.py --skip-install --skip-version-check --medvram --no-half-vae --skip-torch-cuda-test --no-half --opt-split-attention --api
```
These are the arguments from its `webui-user.sh`, plus `--skip-install --skip-version-check` so nothing is updated.
On this install, the "Euler a" sampler produces pure noise on MPS (other samplers are fine); that is the install, not sdmux.

## ComfyUI (existing, unchanged)

Start it with comfy-cli (or the comfy-mcp `launch_comfyui` tool):
```bash
comfy launch --background
```
```bash
comfy stop
```

## Forge (lllyasviel/stable-diffusion-webui-forge)

Installed with:
```bash
cd /Volumes/980Pro/Workspace/AI && git clone --depth 1 https://github.com/lllyasviel/stable-diffusion-webui-forge.git forge
cd forge && /opt/homebrew/bin/python3.10 -m venv venv
venv/bin/python -m pip install "setuptools<70" wheel packaging
venv/bin/python -m pip install --no-build-isolation https://github.com/openai/CLIP/archive/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1.zip
TORCH_COMMAND="pip install torch==2.3.1 torchvision==0.18.1" venv/bin/python launch.py --exit --skip-torch-cuda-test --no-download-sd-model
venv/bin/python -m pip install "numpy==1.26.2" "opencv-contrib-python<4.11" "opencv-python<4.11"
```
The CLIP wheel does not build under new setuptools, and an extension pulls opencv 5 (numpy 2), which breaks
scikit-image; the last two lines fix both. `webui.sh` is not used because `webui-macos-env.sh` sets
`install_dir=$HOME`, which would clone a second copy into the home folder.

Start:
```bash
cd /Volumes/980Pro/Workspace/AI/forge
venv/bin/python launch.py --skip-install --skip-torch-cuda-test --upcast-sampling --no-half-vae --api --port 7863 --no-download-sd-model --ckpt-dir /Volumes/980Pro/Workspace/AI/stable-diffusion-webui/models/Stable-diffusion --lora-dir /Volumes/980Pro/Workspace/AI/stable-diffusion-webui/models/Lora
```

## reForge (Panchovix/stable-diffusion-webui-reForge)

Installed with:
```bash
cd /Volumes/980Pro/Workspace/AI && git clone --depth 1 https://github.com/Panchovix/stable-diffusion-webui-reForge.git reforge
cd reforge && /opt/homebrew/bin/python3.10 -m venv venv
TORCH_COMMAND="pip install torch==2.4.0 torchvision==0.19.0" venv/bin/python launch.py --exit --skip-torch-cuda-test --no-download-sd-model
```
Start:
```bash
cd /Volumes/980Pro/Workspace/AI/reforge
venv/bin/python launch.py --skip-install --skip-torch-cuda-test --upcast-sampling --no-half-vae --api --port 7861 --no-download-sd-model --ckpt-dir /Volumes/980Pro/Workspace/AI/stable-diffusion-webui/models/Stable-diffusion --lora-dir /Volumes/980Pro/Workspace/AI/stable-diffusion-webui/models/Lora
```

## SD.Next (vladmandic/sdnext)

Installed with:
```bash
cd /Volumes/980Pro/Workspace/AI && git clone --depth 1 https://github.com/vladmandic/sdnext.git sdnext
cd sdnext && uv venv --seed --python 3.12 venv && VIRTUAL_ENV=$PWD/venv uv pip install setuptools wheel
mkdir -p models/Stable-diffusion models/Lora
for f in /Volumes/980Pro/Workspace/AI/stable-diffusion-webui/models/Stable-diffusion/*.safetensors; do ln -sf "$f" models/Stable-diffusion/; done
for f in /Volumes/980Pro/Workspace/AI/stable-diffusion-webui/models/Lora/*.safetensors; do ln -sf "$f" models/Lora/; done
```
SD.Next installs its requirements on first start. It has no `--ckpt-dir`; it reads `models/`.
Start (the API is always on):
```bash
cd /Volumes/980Pro/Workspace/AI/sdnext && venv/bin/python launch.py --uv --api-log --port 7862
```
It starts without a model loaded; sdmux switches models through `/sdapi/v1/options`.

## SwarmUI (mcmonkeyprojects/SwarmUI)

.NET 8 SDK installed into a folder, not system-wide:
```bash
curl -sSL -o /Volumes/980Pro/Workspace/AI/.caches/downloads/dotnet/dotnet-install.sh https://dot.net/v1/dotnet-install.sh
bash /Volumes/980Pro/Workspace/AI/.caches/downloads/dotnet/dotnet-install.sh --channel 8.0 --install-dir /Volumes/980Pro/Workspace/AI/dotnet --no-path
```
SwarmUI built from source (it still targets net8.0; it warns that .NET 10 will be required in a future version):
```bash
cd /Volumes/980Pro/Workspace/AI && git clone --depth 1 https://github.com/mcmonkeyprojects/SwarmUI.git SwarmUI
cd SwarmUI && dotnet build src/SwarmUI.csproj --configuration Release -o ./src/bin/live_release && git rev-parse HEAD > src/bin/last_build
```
Configured without the install wizard:
- `Data/Settings.fds`: `IsInstalled: true`, `LaunchMode: none`, `Paths.ModelRoot: /Volumes/980Pro/Workspace/AI/SwarmUI/Models`,
  `Metadata.EditMetadataWriteJSON: true`, `Metadata.ImageMetadataIncludeModelHash: false`
  (so SwarmUI keeps its metadata in `.swarm.json` files instead of rewriting model files).
- `Data/Backends.fds`: one `comfyui_api` backend at `http://127.0.0.1:8188` (the existing ComfyUI).
- `Models/Stable-Diffusion` and `Models/Lora`: per-file symlinks to A1111's and ComfyUI's checkpoints and A1111's LoRAs.

Start ComfyUI first, then:
```bash
cd /Volumes/980Pro/Workspace/AI/SwarmUI && ASPNETCORE_ENVIRONMENT=Production ./src/bin/live_release/SwarmUI --launch_mode none --port 7801
```
The existing ComfyUI does not have SwarmUI's own nodes (`SwarmComfyCommon`), so SwarmUI warns "missing the Swarm core
nodes". Plain txt2img, img2img and inpaint work; mask blur and some advanced features need those nodes. To add them
without touching the ComfyUI install, launch ComfyUI with an extra `--extra-model-paths-config` file whose
`custom_nodes` entry points at `SwarmUI/src/BuiltinExtensions/ComfyUIBackend/ExtraNodes` (not done here).

## InvokeAI

```bash
uv venv --python 3.12 /Volumes/980Pro/Workspace/AI/InvokeAI/.venv
VIRTUAL_ENV=/Volumes/980Pro/Workspace/AI/InvokeAI/.venv uv pip install invokeai==6.14.4
```
Start:
```bash
INVOKEAI_ROOT=/Volumes/980Pro/Workspace/AI/InvokeAI /Volumes/980Pro/Workspace/AI/InvokeAI/.venv/bin/invokeai-web --root /Volumes/980Pro/Workspace/AI/InvokeAI
```
Models registered in place (InvokeAI hashes them and records the path; it does not copy or modify them):
```bash
curl -X POST -H 'Content-Type: application/json' -d '{}' "http://127.0.0.1:9090/api/v2/models/install?inplace=true&source=/Volumes/980Pro/Workspace/AI/stable-diffusion-webui/models/Stable-diffusion/hassakuSD15_v13.safetensors"
```
Registered: `hassakuSD15_v13`, `meinahentai_v4` (main) and `2b_nierautomata-10` (LoRA).

## Incident during setup (2026-11-12)

Before the per-file symlinks, SwarmUI rewrote the headers of `hassakuSD15_v13.safetensors` and the LoRA
`2b_nierautomata-10.safetensors` in A1111's folders, and SD.Next created four `.thumb.jpg` files there. Both headers
were rebuilt and verified byte-for-byte against the sha256 recorded by A1111's hash cache and the LoRA Manager sidecar,
their modification times restored, and the thumbnails deleted.
