"""
ComfyUI, through its HTTP API (/prompt, /history, /view) and optionally its websocket for step progress.

Each request becomes a small API-format workflow built from core nodes. Node inputs
are filled from the server's /object_info, so required inputs added by newer
ComfyUI versions get their defaults instead of failing validation.

Native escape hatch: run_workflow(workflow) queues any API-format workflow and
returns its output images; queue(), wait() and fetch() are the steps it is made of.
"""

from __future__ import annotations

import hashlib
import io
import json
import posixpath
import threading
import time
import uuid
from typing import Any

from PIL import Image

from ..errors import Cancelled, NotReady, Rejected, SdmuxError, TimedOut, Unsupported
from ..names import canonical_scheduler, find_choice, pick_native, sampler_id
from ..types import INPAINT, PIXEL_UPSCALE, TXT2IMG, Model, Progress, Request, Result, infotext, load_image, to_png_bytes
from .base import Backend


def combo_options(spec) -> list[str]:
    """Options of a combo input from /object_info, in the old list format or the newer COMBO format."""
    if not spec:
        return []
    first = spec[0]
    if isinstance(first, list):
        return list(first)
    if first == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
        return list(spec[1].get("options", []))
    return []


def input_default(spec):
    """Default value of an input from /object_info; None for links and inputs without one."""
    options = combo_options(spec)
    if options:
        return options[0]
    if len(spec) > 1 and isinstance(spec[1], dict) and "default" in spec[1]:
        return spec[1]["default"]
    if spec and spec[0] == "STRING":
        return ""
    return None


# Split-file loaders per architecture: which CLIP loader and type, which text encoders to look for.
# Used when the model is listed by UNETLoader (models/diffusion_models) rather than CheckpointLoaderSimple.
SPLIT_FILE: dict[str, dict[str, Any]] = {
    "flux": {"clip": ("DualCLIPLoader", "flux", ("clip_l", "t5xxl")), "vae": ("ae", "flux"), "guidance": 3.5, "cfg": 1.0},
    "flux2": {"clip": ("CLIPLoader", "flux2", ("mistral", "qwen")), "vae": ("flux2",), "guidance": 4.0, "cfg": 1.0},
    "chroma": {"clip": ("CLIPLoader", "chroma", ("t5xxl",)), "vae": ("ae", "flux"), "guidance": None, "cfg": 4.0},
    "qwen-image": {"clip": ("CLIPLoader", "qwen_image", ("qwen",)), "vae": ("qwen_image",), "guidance": None, "cfg": 2.5},
    "sd3": {"clip": ("TripleCLIPLoader", None, ("clip_l", "clip_g", "t5xxl")), "vae": ("sd3",), "guidance": None, "cfg": 4.5},
    "lumina2": {"clip": ("CLIPLoader", "lumina2", ("gemma",)), "vae": ("ae", "flux"), "guidance": None, "cfg": 4.0},
}


class ComfyUI(Backend):
    kind = "comfyui"
    label = "ComfyUI"
    default_url = "http://127.0.0.1:8188"

    def __init__(self, url=None, options=None, **kw) -> None:
        super().__init__(url, options, **kw)
        self.client_id = uuid.uuid4().hex
        self._object_info: dict[str, dict] = {}
        self._uploaded: dict[str, str] = {}
        self._prompt_id: str | None = None
        self._progress: Progress | None = None
        self._folders: dict[str, list[str]] | None = None
        self._all_nodes: dict | None = None

    # Server info ----------------------------------------------------------
    def object_info(self, node: str, refresh: bool = False) -> dict:
        """The /object_info entry of a node class, {} when the server does not have it."""
        if refresh or node not in self._object_info:
            response = self.http.request("GET", f"/object_info/{node}")
            if response.status_code == 404:
                return {}
            info = self.http.json(self.http.check(response, f"GET /object_info/{node}"))
            if node not in info:
                return {}
            self._object_info[node] = info[node]
        return self._object_info[node]

    # Node contract --------------------------------------------------------
    def nodes(self, refresh: bool = False) -> list:
        """Every node the server has loaded (custom packs included), as NodeSpecs sorted by name."""
        from ..comfygraph import parse_spec

        if refresh or self._all_nodes is None:
            info = self.http.get_json("/object_info", timeout=max(self.options.timeout, 120))
            self._all_nodes = {name: parse_spec(name, entry) for name, entry in info.items()}
            self._object_info.update(info)
        return [self._all_nodes[k] for k in sorted(self._all_nodes)]

    def node(self, name: str):
        """The typed spec of one node (inputs with types, defaults, ranges, options; named outputs), None if missing."""
        from ..comfygraph import parse_spec

        if self._all_nodes is not None and name in self._all_nodes:
            return self._all_nodes[name]
        info = self.object_info(name)
        return parse_spec(name, info) if info else None

    def graph(self, workflow: dict | None = None):
        """A new Graph checked against this server's nodes, or one wrapping an existing API-format workflow."""
        from ..comfygraph import Graph

        return Graph.from_api(workflow, self.node) if workflow is not None else Graph(self.node)

    def run(self, graph, validate: bool = True, timeout: float | None = None) -> Result:
        """
        Validate a Graph against the node contract and run it. Result.images holds every output image;
        Result.raw["outputs"] maps each output node id to its images.
        """
        if validate:
            graph.validate()
        workflow = graph.to_api()
        result = self.run_workflow(workflow, timeout)
        entry = result.raw["history"]
        result.raw["outputs"] = {node_id: self.fetch(entry, node_id) for node_id in (entry.get("outputs") or {})}
        return result

    def workflow(self, request: Request | None = None, **kwargs):
        """
        The Graph sdmux would run for a request (or txt2img keyword arguments), without running it.
        Images in an img2img/inpaint request are uploaded so the graph is runnable.
        """
        request = request or Request(**kwargs)
        request = self._prepare(request)
        self._fill_defaults(request)
        request.seed = request.resolved_seed()
        image_name = self.upload(request.image) if request.image is not None else None
        mask_name = self.upload(request.mask, "mask") if request.mask is not None else None
        return self.graph(self.build_workflow(request, image_name, mask_name))

    def options_for(self, node: str, name: str, refresh: bool = False) -> list[str]:
        inputs = self.object_info(node, refresh).get("input", {})
        spec = inputs.get("required", {}).get(name) or inputs.get("optional", {}).get(name)
        return combo_options(spec)

    def refresh(self) -> None:
        """Forget cached node info; ComfyUI rescans model folders on the next /object_info call."""
        self._object_info.clear()
        self._all_nodes = None
        self._folders = None

    def _model_folders(self) -> dict[str, list[str]]:
        if self._folders is None:
            response = self.http.request("GET", "/experiment/models", timeout=10)
            folders = {}
            if response.status_code == 200:
                try:
                    folders = {f["name"]: f.get("folders") or [] for f in response.json()}
                except (ValueError, KeyError, TypeError):
                    folders = {}
            self._folders = folders
        return self._folders

    def _paths(self, folder: str) -> dict[str, str]:
        """Listed name -> absolute path as the server reports it (may not exist on this machine)."""
        roots = self._model_folders().get(folder) or []
        if not roots:
            return {}
        response = self.http.request("GET", f"/experiment/models/{folder}", timeout=30)
        if response.status_code != 200:
            return {}
        paths = {}
        try:
            for item in response.json():
                index = item.get("pathIndex", 0)
                if 0 <= index < len(roots):
                    paths[item["name"]] = posixpath.join(roots[index].replace("\\", "/"), item["name"])
        except (ValueError, KeyError, TypeError):
            return {}
        return paths

    def _listing(self, node: str, field: str, folder: str, kind: str, refresh: bool = True) -> list[Model]:
        names = self.options_for(node, field, refresh=refresh)
        paths = self._paths(folder) if names else {}
        return [Model(name=n, path=paths.get(n), kind=kind) for n in names]

    def models(self) -> list[Model]:
        """Single-file checkpoints (models/checkpoints) followed by diffusion models (models/diffusion_models)."""
        models = self._listing("CheckpointLoaderSimple", "ckpt_name", "checkpoints", "checkpoint")
        models += self._listing("UNETLoader", "unet_name", "diffusion_models", "diffusion_model")
        return models

    def loras(self) -> list[Model]:
        return self._listing("LoraLoader", "lora_name", "loras", "lora")

    def samplers(self) -> list[str]:
        return self.options_for("KSampler", "sampler_name")

    def schedulers(self) -> list[str]:
        return self.options_for("KSampler", "scheduler")

    def info(self) -> dict:
        stats = self.http.get_json("/system_stats", timeout=10)
        system = stats.get("system", {})
        return {
            "backend": self.kind,
            "url": self.url,
            "version": system.get("comfyui_version"),
            "python": system.get("python_version"),
            "devices": [d.get("name") for d in stats.get("devices", [])],
        }

    def ready(self) -> dict:
        problems = []
        info = self.info()
        self.refresh()
        for node in ("CheckpointLoaderSimple", "KSampler", "CLIPTextEncode", "VAEDecode"):
            if not self.object_info(node):
                problems.append(f"ComfyUI has no {node} node.")
        if not self.models():
            problems.append("ComfyUI lists no checkpoints or diffusion models.")
        return {"ready": not problems, "problems": problems, "info": info}

    # Workflow helpers -----------------------------------------------------
    def _node(self, class_type: str, inputs: dict) -> dict:
        """A workflow node, with required inputs the server declares but were not given set to their defaults."""
        info = self.object_info(class_type).get("input", {})
        declared = {**info.get("required", {}), **info.get("optional", {})}
        if declared:
            inputs = {k: v for k, v in inputs.items() if k in declared}
            for name, spec in info.get("required", {}).items():
                if name not in inputs:
                    default = input_default(spec)
                    if default is not None:
                        inputs[name] = default
        return {"class_type": class_type, "inputs": inputs}

    def _pick(self, value: str | None, node: str, field: str, canonical, fallback: str) -> str:
        choices = self.options_for(node, field)
        if value:
            native = pick_native(value, choices, canonical)
            if native:
                return native
            raise Rejected(f"ComfyUI has no {field} matching {value!r}. It lists: {', '.join(choices)}", backend=self.kind)
        return fallback if not choices or fallback in choices else choices[0]

    def _find_file(self, node: str, field: str, hints: tuple[str, ...], what: str) -> str:
        choices = self.options_for(node, field)
        for hint in hints:
            for choice in choices:
                if hint.lower() in choice.lower():
                    return choice
        raise NotReady(f"ComfyUI has no {what} ({' or '.join(hints)}) for {node}.{field}.", backend=self.kind)

    def _loaders(self, workflow: dict, request: Request, model: Model) -> tuple[list, list, list, dict[str, Any]]:
        """Add the model loaders; returns (model, clip, vae) links and sampling hints."""
        if model.kind == "checkpoint":
            workflow["ckpt"] = self._node("CheckpointLoaderSimple", {"ckpt_name": model.name})
            links = (["ckpt", 0], ["ckpt", 1], ["ckpt", 2])
            if request.vae:
                vae = self._pick(request.vae, "VAELoader", "vae_name", lambda s: s.lower(), "")
                workflow["vae"] = self._node("VAELoader", {"vae_name": vae})
                links = (links[0], links[1], ["vae", 0])
            return (*links, {})

        # Split-file model: needs to know the architecture to pick text encoders and VAE.
        arch = model.architecture or self.identify(model).architecture
        spec = SPLIT_FILE.get(arch or "")
        if not spec:
            raise Unsupported(
                f"sdmux cannot build a ComfyUI graph for the diffusion model {model.name} "
                f"(architecture {arch or 'unknown'}). Use run_workflow() with your own graph.",
                backend=self.kind,
            )
        workflow["unet"] = self._node("UNETLoader", {"unet_name": model.name, "weight_dtype": "default"})
        loader, clip_type, encoders = spec["clip"]
        if loader == "DualCLIPLoader":
            names = {"clip_name1": self._find_file(loader, "clip_name1", (encoders[0],), "text encoder"),
                     "clip_name2": self._find_file(loader, "clip_name2", (encoders[1],), "text encoder")}
        elif loader == "TripleCLIPLoader":
            names = {f"clip_name{i + 1}": self._find_file(loader, f"clip_name{i + 1}", (enc,), "text encoder")
                     for i, enc in enumerate(encoders)}
        else:
            names = {"clip_name": self._find_file(loader, "clip_name", encoders, "text encoder")}
        clip_inputs = dict(names)
        if clip_type:
            clip_inputs["type"] = clip_type
        workflow["clip"] = self._node(loader, clip_inputs)
        vae_name = request.vae or self._find_file("VAELoader", "vae_name", spec["vae"], "VAE")
        workflow["vae"] = self._node("VAELoader", {"vae_name": vae_name})
        return ["unet", 0], ["clip", 0], ["vae", 0], spec

    def build_workflow(self, request: Request, image_name: str | None = None, mask_name: str | None = None) -> dict:
        """The API-format workflow for a request. Images must already be uploaded (see upload())."""
        model = self.resolve_model(request.model)
        prompt, loras = self.resolve_loras(request.prompt)
        workflow: dict = {}
        model_link, clip_link, vae_link, hints = self._loaders(workflow, request, model)

        for i, (lora, weight) in enumerate(loras):
            node_id = f"lora_{i}"
            if model.kind == "checkpoint":
                workflow[node_id] = self._node("LoraLoader", {"model": model_link, "clip": clip_link, "lora_name": lora.name,
                                                             "strength_model": weight, "strength_clip": weight})
                model_link, clip_link = [node_id, 0], [node_id, 1]
            else:
                workflow[node_id] = self._node("LoraLoaderModelOnly", {"model": model_link, "lora_name": lora.name,
                                                                      "strength_model": weight})
                model_link = [node_id, 0]

        if request.clip_skip and request.clip_skip > 1 and model.kind == "checkpoint":
            workflow["clip_skip"] = self._node("CLIPSetLastLayer", {"clip": clip_link, "stop_at_clip_layer": -int(request.clip_skip)})
            clip_link = ["clip_skip", 0]

        workflow["positive"] = self._node("CLIPTextEncode", {"text": prompt, "clip": clip_link})
        workflow["negative"] = self._node("CLIPTextEncode", {"text": request.negative, "clip": clip_link})
        positive = ["positive", 0]
        if hints.get("guidance") is not None:
            workflow["guidance"] = self._node("FluxGuidance", {"conditioning": positive, "guidance": hints["guidance"]})
            positive = ["guidance", 0]

        width, height = request.size()
        if request.mode == TXT2IMG:
            latent_node = "EmptySD3LatentImage" if model.kind != "checkpoint" and self.object_info("EmptySD3LatentImage") else "EmptyLatentImage"
            workflow["latent"] = self._node(latent_node, {"width": width, "height": height, "batch_size": request.batch_size})
            latent = ["latent", 0]
        else:
            workflow["image"] = self._node("LoadImage", {"image": image_name})
            pixels = ["image", 0]
            if (width, height) != request.image.size:
                workflow["scaled"] = self._node("ImageScale", {"image": pixels, "upscale_method": "lanczos",
                                                              "width": width, "height": height, "crop": "disabled"})
                pixels = ["scaled", 0]
            workflow["encode"] = self._node("VAEEncode", {"pixels": pixels, "vae": vae_link})
            latent = ["encode", 0]
            if request.mode == INPAINT:
                workflow["mask"] = self._node("LoadImageMask", {"image": mask_name, "channel": "red"})
                mask = ["mask", 0]
                if request.mask.size != (width, height):
                    workflow["mask_image"] = self._node("MaskToImage", {"mask": mask})
                    workflow["mask_scaled"] = self._node("ImageScale", {"image": ["mask_image", 0], "upscale_method": "bilinear",
                                                                       "width": width, "height": height, "crop": "disabled"})
                    workflow["mask_resized"] = self._node("ImageToMask", {"image": ["mask_scaled", 0], "channel": "red"})
                    mask = ["mask_resized", 0]
                workflow["noise_mask"] = self._node("SetLatentNoiseMask", {"samples": latent, "mask": mask})
                latent = ["noise_mask", 0]
            if request.batch_size > 1:
                workflow["batch"] = self._node("RepeatLatentBatch", {"samples": latent, "amount": request.batch_size})
                latent = ["batch", 0]

        cfg = request.cfg
        workflow["sampler"] = self._node(
            "KSampler",
            {
                "model": model_link,
                "seed": request.seed,
                "steps": request.steps,
                "cfg": cfg,
                "sampler_name": self._pick(request.sampler, "KSampler", "sampler_name", sampler_id, "euler"),
                "scheduler": self._pick(request.scheduler or _folded_scheduler(request.sampler), "KSampler", "scheduler",
                                        canonical_scheduler, "normal"),
                "positive": positive,
                "negative": ["negative", 0],
                "latent_image": latent,
                "denoise": 1.0 if request.mode == TXT2IMG else min(max(float(request.denoise), 0.0), 1.0),
            },
        )
        samples = ["sampler", 0]
        if request.hires is not None:
            samples = self._hires(workflow, request, samples, vae_link)
        workflow["decode"] = self._node("VAEDecode", {"samples": samples, "vae": vae_link})
        output = ["decode", 0]
        if request.mode == INPAINT:
            # Keep pixels outside the mask exactly as they were, as A1111 does.
            base = workflow.get("scaled") and ["scaled", 0] or ["image", 0]
            workflow["composite"] = self._node("ImageCompositeMasked", {"destination": base, "source": output, "x": 0, "y": 0,
                                                                       "resize_source": False, "mask": workflow["noise_mask"]["inputs"]["mask"]})
            output = ["composite", 0]
        workflow["output"] = {"class_type": "PreviewImage", "inputs": {"images": output}}
        return workflow

    def _hires(self, workflow: dict, request: Request, samples: list, vae_link: list) -> list:
        """Upscale the first pass (latent, pixels or an upscale model) and sample it again; returns the new latent."""
        hires = request.hires
        width, height = hires.size(*request.size())
        if hires.latent_method:
            method = self._pick(hires.latent_method, "LatentUpscale", "upscale_method", str.lower, "nearest-exact")
            workflow["hires_upscale"] = self._node("LatentUpscale", {"samples": samples, "upscale_method": method,
                                                                     "width": width, "height": height, "crop": "disabled"})
        else:
            workflow["hires_pixels"] = self._node("VAEDecode", {"samples": samples, "vae": vae_link})
            pixels = ["hires_pixels", 0]
            method = hires.pixel_method
            if method is None:
                assert hires.upscaler is not None  # a model name
                workflow["hires_model"] = self._node("UpscaleModelLoader", {"model_name": self._upscale_model(hires.upscaler)})
                workflow["hires_model_upscaled"] = self._node("ImageUpscaleWithModel", {"upscale_model": ["hires_model", 0],
                                                                                        "image": pixels})
                pixels, method = ["hires_model_upscaled", 0], "lanczos"
            workflow["hires_scaled"] = self._node("ImageScale", {"image": pixels, "upscale_method": method,
                                                                 "width": width, "height": height, "crop": "disabled"})
            workflow["hires_upscale"] = self._node("VAEEncode", {"pixels": ["hires_scaled", 0], "vae": vae_link})
        second = dict(workflow["sampler"]["inputs"])
        second.update(latent_image=["hires_upscale", 0], steps=hires.steps or request.steps,
                      denoise=min(max(float(hires.denoise), 0.0), 1.0))
        workflow["hires_sampler"] = self._node("KSampler", second)
        return ["hires_sampler", 0]

    def _upscale_model(self, wanted: str) -> str:
        names = self.upscalers()
        found = find_choice(wanted, names)
        if not found:
            raise Rejected(f"ComfyUI has no upscale model matching {wanted!r}. It lists: {', '.join(names) or 'none'}",
                           backend=self.kind)
        return found

    # Upscaling ------------------------------------------------------------
    def upscalers(self) -> list[str]:
        """Upscale models in models/upscale_models."""
        return self.options_for("UpscaleModelLoader", "model_name")

    def upscale(self, image, upscaler: str | None = None, scale: float = 2.0) -> Result:
        """Upscale with a model from upscale_models (or a plain resize: "Lanczos", "Bicubic", ...) to scale x the size."""
        source = load_image(image)
        width, height = int(source.width * scale), int(source.height * scale)
        workflow: dict = {"image": self._node("LoadImage", {"image": self.upload(source)})}
        pixels, method = ["image", 0], PIXEL_UPSCALE.get((upscaler or "lanczos").strip().lower())
        if method is None:
            assert upscaler is not None
            workflow["model"] = self._node("UpscaleModelLoader", {"model_name": self._upscale_model(upscaler)})
            workflow["upscaled"] = self._node("ImageUpscaleWithModel", {"upscale_model": ["model", 0], "image": pixels})
            pixels, method = ["upscaled", 0], "lanczos"
        workflow["scaled"] = self._node("ImageScale", {"image": pixels, "upscale_method": method, "width": width,
                                                       "height": height, "crop": "disabled"})
        workflow["output"] = {"class_type": "PreviewImage", "inputs": {"images": ["scaled", 0]}}
        entry = self._execute(workflow)
        images = self.fetch(entry, "output")
        if not images:
            raise Rejected("ComfyUI finished without an output image.", backend=self.kind, details=entry)
        return Result(images=images, backend=self.kind, raw={"history": entry, "workflow": workflow})

    # Running --------------------------------------------------------------
    def upload(self, image: Image.Image, kind: str = "image") -> str:
        """Upload a PIL image to ComfyUI's input folder (once per content) and return the name LoadImage uses."""
        data = to_png_bytes(image.convert("RGB") if kind == "mask" else image)
        digest = hashlib.sha1(data).hexdigest()[:16]
        if digest in self._uploaded:
            return self._uploaded[digest]
        response = self.http.request(
            "POST", "/upload/image",
            files={"image": (f"sdmux_{digest}.png", data, "image/png")},
            data={"type": "input", "overwrite": "true"},
            timeout=120,
        )
        result = self.http.json(self.http.check(response, "image upload"))
        name = f"{result['subfolder']}/{result['name']}" if result.get("subfolder") else result["name"]
        self._uploaded[digest] = name
        return name

    def queue(self, workflow: dict) -> str:
        """POST /prompt; returns the prompt id. Validation errors raise Rejected with one line per failing input."""
        response = self.http.request("POST", "/prompt", json={"prompt": workflow, "client_id": self.client_id})
        if response.status_code != 200:
            raise Rejected(f"ComfyUI rejected the workflow: {describe_rejection(response)}", backend=self.kind,
                           details=response.text)
        return self.http.json(response)["prompt_id"]

    def wait(self, prompt_id: str, timeout: float | None = None) -> dict:
        """Poll /history until the prompt finishes; return its history entry."""
        deadline = time.monotonic() + (timeout or self.options.job_timeout)
        while time.monotonic() < deadline:
            history = self.http.get_json(f"/history/{prompt_id}")
            entry = history.get(prompt_id)
            if entry:
                status = entry.get("status") or {}
                messages = status.get("messages") or []
                if status.get("status_str") == "error" or (status and not status.get("completed", True)):
                    if any(m and m[0] == "execution_interrupted" for m in messages) or self._cancel.is_set():
                        raise Cancelled("Interrupted.", backend=self.kind)
                    details = [m[1] for m in messages if m and m[0] == "execution_error" and len(m) > 1]
                    detail = details[0] if details else {}
                    raise Rejected(
                        f"ComfyUI failed in {detail.get('node_type', 'the workflow')}: "
                        f"{str(detail.get('exception_message', 'unknown error')).strip()}",
                        backend=self.kind, details=detail,
                    )
                return entry
            if self._cancel.is_set():
                raise Cancelled("Interrupted.", backend=self.kind)
            time.sleep(self.options.poll_interval)
        try:
            self.cancel(prompt_id)
        except SdmuxError:
            pass
        raise TimedOut(f"ComfyUI job {prompt_id} did not finish within {timeout or self.options.job_timeout:.0f}s.",
                       backend=self.kind)

    def cancel(self, prompt_id: str) -> None:
        """Drop prompt_id from the queue if it is pending, interrupt it if it is running; leave other jobs alone."""
        queue = self.http.get_json("/queue", timeout=10)
        running = {item[1] for item in queue.get("queue_running") or [] if len(item) > 1}
        pending = {item[1] for item in queue.get("queue_pending") or [] if len(item) > 1}
        if prompt_id in pending:
            self.http.check(self.http.request("POST", "/queue", json={"delete": [prompt_id]}, timeout=10), "queue delete")
        if prompt_id in running:
            self.http.check(self.http.request("POST", "/interrupt", json={"prompt_id": prompt_id}, timeout=10), "interrupt")

    def fetch(self, entry: dict, node: str | None = None) -> list[Image.Image]:
        """Download the images a history entry lists, for one output node or all of them."""
        outputs = entry.get("outputs") or {}
        items: list[dict] = []
        for node_id, out in outputs.items():
            if node is None or node_id == node:
                items.extend(out.get("images") or [])
        images: list[Image.Image] = []
        for item in items:
            response = self._view({"filename": item["filename"], "subfolder": item.get("subfolder", ""),
                                   "type": item.get("type", "output")})
            self.http.check(response, "image download")
            image = Image.open(io.BytesIO(response.content))
            image.load()
            images.append(image)
        return images

    # A proxy in front of ComfyUI (RunPod's, for one) now and then answers a download with a 404 or a
    # 5xx although the file is there: try again a few times before giving up on the job.
    VIEW_TRIES = 4

    def _view(self, params: dict):
        for attempt in range(self.VIEW_TRIES):
            response = self.http.request("GET", "/view", params=params, timeout=120)
            if response.status_code != 404 and response.status_code < 500:
                return response
            if attempt + 1 < self.VIEW_TRIES:
                time.sleep(min(8.0, 1.0 * 2**attempt))
        return response

    def run_workflow(self, workflow: dict, timeout: float | None = None) -> Result:
        """
        Queue any API-format workflow (as exported with "Save (API)") and return every output image.

        Raises Rejected on validation or execution errors, Cancelled on interrupt, TimedOut past the job timeout.
        """
        entry = self._execute(workflow, timeout)
        images = self.fetch(entry)
        if not images:
            raise Rejected("ComfyUI finished without an output image.", backend=self.kind, details=entry)
        return Result(images=images, backend=self.kind, raw={"history": entry, "workflow": workflow})

    def _execute(self, workflow: dict, timeout: float | None = None) -> dict:
        with self._job():
            stop = self._listen()
            try:
                self._prompt_id = self.queue(workflow)
                if self._cancel.is_set():
                    # interrupt() arrived before the prompt id was known; stop just this prompt.
                    self.cancel(self._prompt_id)
                return self.wait(self._prompt_id, timeout)
            finally:
                if stop:
                    stop.set()
                self._prompt_id = None
                self._progress = None

    def _generate(self, request: Request) -> Result:
        request.seed = request.resolved_seed()
        image_name = self.upload(request.image) if request.image is not None else None
        mask_name = self.upload(request.mask, "mask") if request.mask is not None else None
        workflow = self.build_workflow(request, image_name, mask_name)
        if request.patch is not None:
            # The caller's hook edits the built graph: add nodes, rewire, change inputs. Node "output" is returned.
            graph = self.graph(workflow)
            request.patch(graph)
            graph.validate()
            workflow = graph.to_api()
        entry = self._execute(workflow)
        images = self.fetch(entry, "output")
        if not images:
            raise Rejected("ComfyUI finished without an output image.", backend=self.kind, details=entry)
        sampler = (workflow.get("sampler") or {}).get("inputs") or {}
        loader = workflow.get("ckpt") or workflow.get("unet")
        model_name = (loader["inputs"].get("ckpt_name") or loader["inputs"].get("unet_name")) if loader else None
        text = infotext(request, seed=request.seed, size=images[0].size, sampler=sampler.get("sampler_name"),
                        scheduler=sampler.get("scheduler"), model=model_name, version="ComfyUI")
        for image in images:
            image.info["parameters"] = text
        return Result(images=images, seed=request.seed, seeds=[request.seed] * len(images), parameters=text,
                      backend=self.kind, raw={"history": entry, "workflow": workflow})

    # Progress -------------------------------------------------------------
    def _listen(self) -> threading.Event | None:
        """Follow step progress on the websocket (needs websocket-client). Returns a stop event, or None."""
        try:
            import websocket  # websocket-client, optional
        except ImportError:
            return None
        ws_url = "ws" + self.url[len("http"):] + f"/ws?clientId={self.client_id}"
        headers = [f"{k}: {v}" for k, v in self.http.auth_headers().items()]
        try:
            ws = websocket.create_connection(ws_url, timeout=self.options.connect_timeout, header=headers,
                                             sslopt=None if self.options.verify_tls else {"cert_reqs": 0})
        except Exception:  # progress is optional, never fail the job over it
            return None
        stop = threading.Event()

        def run():
            ws.settimeout(1)
            try:
                while not stop.is_set():
                    try:
                        message = ws.recv()
                    except websocket.WebSocketTimeoutException:
                        continue
                    if not isinstance(message, str):
                        continue  # binary preview frames
                    event = json.loads(message)
                    data = event.get("data") or {}
                    if event.get("type") == "progress" and data.get("max") and data.get("prompt_id") in (None, self._prompt_id):
                        self._progress = Progress(fraction=data["value"] / data["max"], step=data["value"],
                                                  steps=data["max"], active=True)
            except Exception:
                pass
            finally:
                ws.close()

        threading.Thread(target=run, name="sdmux-comfyui-progress", daemon=True).start()
        return stop

    def progress(self) -> Progress:
        if self._progress is not None:
            return self._progress
        queue = self.http.get_json("/queue", timeout=10)
        running = queue.get("queue_running") or []
        return Progress(active=bool(running))

    def interrupt(self) -> None:
        """
        Stop this client's prompt: drop it from the queue if it has not started, interrupt it if it has.
        With no job of this client in flight, interrupt whatever the server is running.
        """
        self._cancel.set()
        prompt_id = self._prompt_id
        if prompt_id:
            self.cancel(prompt_id)
        elif not self._lock.locked():
            self.http.check(self.http.request("POST", "/interrupt", timeout=10), "interrupt")
        # Otherwise a job is being queued; it sees the cancel flag and stops its own prompt.

    # Identification -------------------------------------------------------
    def metadata(self, model: str | Model, folder: str = "checkpoints") -> dict:
        """A model file's safetensors __metadata__ as the server reads it (works remotely), {} when none."""
        name = model.name if isinstance(model, Model) else model
        response = self.http.request("GET", f"/view_metadata/{folder}", params={"filename": name}, timeout=30)
        if response.status_code != 200:
            return {}
        try:
            data = response.json()
        except ValueError:
            return {}
        return data if isinstance(data, dict) else {}


def _folded_scheduler(sampler: str | None) -> str | None:
    from ..names import canonical_sampler

    return canonical_sampler(sampler)[1] if sampler else None


def describe_rejection(response) -> str:
    """ComfyUI's validation error body as one readable line per failing node input."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:2000]
    lines = []
    error = body.get("error") or {}
    if isinstance(error, dict) and error.get("message"):
        lines.append(error["message"])
    for node_id, node in (body.get("node_errors") or {}).items():
        for err in node.get("errors", []):
            lines.append(f"{node.get('class_type', node_id)}: {err.get('message')} {err.get('details', '')}".strip())
    return "; ".join(lines) or response.text[:2000]
