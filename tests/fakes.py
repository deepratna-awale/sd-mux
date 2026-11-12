"""Small fake HTTP servers, one per API family, for offline tests. They speak just enough of each API."""

from __future__ import annotations

import base64
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from PIL import Image


def png_bytes(color=(200, 50, 50), size=(64, 64), text: dict | None = None) -> bytes:
    from PIL import PngImagePlugin

    image = Image.new("RGB", size, color)
    buffer = io.BytesIO()
    meta = None
    if text:
        meta = PngImagePlugin.PngInfo()
        for k, v in text.items():
            meta.add_text(k, v)
    image.save(buffer, format="PNG", pnginfo=meta)
    return buffer.getvalue()


def png_b64(**kw) -> str:
    return base64.b64encode(png_bytes(**kw)).decode()


class Reply:
    def __init__(self, status=200, body=None, content_type="application/json", delay=0.0, headers=None):
        self.status, self.body, self.content_type, self.delay, self.headers = status, body, content_type, delay, headers or {}


class FakeServer:
    """routes: {(METHOD, path): handler(request) -> Reply | dict | list}. request has .json, .query, .body, .headers, .path."""

    def __init__(self, prefix: str = ""):
        self.routes: dict = {}
        self.requests: list = []
        self.prefix = prefix
        self.require_auth: tuple[str, str] | None = None
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _handle(self, method):
                parts = urlsplit(self.path)
                path = parts.path
                if fake.prefix and path.startswith(fake.prefix):
                    path = path[len(fake.prefix):] or "/"
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                req = type("Req", (), {})()
                req.method, req.path, req.body, req.headers = method, path, body, dict(self.headers)
                req.query = {k: v[0] for k, v in parse_qs(parts.query).items()}
                try:
                    req.json = json.loads(body) if body and "json" in (self.headers.get("Content-Type") or "") else None
                except ValueError:
                    req.json = None
                fake.requests.append(req)
                if fake.require_auth:
                    expected = "Basic " + base64.b64encode(":".join(fake.require_auth).encode()).decode()
                    if self.headers.get("Authorization") != expected:
                        return self._send(Reply(401, {"detail": "Not authenticated"}))
                handler = fake.routes.get((method, path))
                if handler is None:
                    for (m, p), h in fake.routes.items():
                        if m == method and p.endswith("*") and path.startswith(p[:-1]):
                            handler = h
                            break
                if handler is None:
                    return self._send(Reply(404, {"detail": "Not Found"}))
                result = handler(req)
                if not isinstance(result, Reply):
                    result = Reply(200, result)
                self._send(result)

            def _send(self, reply: Reply):
                if reply.delay:
                    time.sleep(reply.delay)
                body = reply.body
                if not isinstance(body, (bytes, bytearray)):
                    body = json.dumps(body).encode()
                try:
                    self.send_response(reply.status)
                    self.send_header("Content-Type", reply.content_type)
                    self.send_header("Content-Length", str(len(body)))
                    for k, v in reply.headers.items():
                        self.send_header(k, v)
                    self.end_headers()
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self):
                self._handle("GET")

            def do_POST(self):
                self._handle("POST")

            def do_PUT(self):
                self._handle("PUT")

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}{self.prefix}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()

    def route(self, method: str, path: str):
        def deco(fn):
            self.routes[(method, path)] = fn
            return fn
        return deco

    def last(self, method: str, path: str):
        return next(r for r in reversed(self.requests) if r.method == method and r.path == path)


# A1111 family -------------------------------------------------------------------
def a1111_server(variant: str = "a1111", prefix: str = "") -> FakeServer:
    s = FakeServer(prefix)
    s.state = {"interrupted": False}
    s.routes[("GET", "/sdapi/v1/samplers")] = lambda r: [{"name": n, "aliases": [], "options": {}} for n in
                                                          ("DPM++ 2M", "Euler a", "Euler", "DDIM")]
    if variant != "sdnext":
        s.routes[("GET", "/sdapi/v1/schedulers")] = lambda r: [{"name": n, "label": l} for n, l in
                                                                (("automatic", "Automatic"), ("karras", "Karras"), ("normal", "Normal"))]
    s.routes[("GET", "/sdapi/v1/sd-models")] = lambda r: [
        {"title": "hassakuSD15_v13.safetensors [7eb674963a]", "model_name": "hassakuSD15_v13", "hash": "7eb674963a",
         "sha256": "7eb6", "filename": "/nowhere/hassakuSD15_v13.safetensors"},
        {"title": "sdxl/waiIllustriousSDXL_v170.safetensors [abc]", "model_name": "sdxl_waiIllustriousSDXL_v170", "hash": "abc",
         "sha256": None, "filename": "/nowhere/sdxl/waiIllustriousSDXL_v170.safetensors"},
    ]
    s.routes[("GET", "/sdapi/v1/loras")] = lambda r: [{"name": "detail", "alias": "detail", "path": "/nowhere/detail.safetensors",
                                                       "metadata": {}}]
    versions = {"a1111": {"app": "stable-diffusion-webui", "version": "v1.10.1"},
                "forge": {"app": "stable-diffusion-webui", "version": "f2.0.1v1.10.1-previous"},
                "sdnext": {"app": "sd.next", "version": "2026-11-01"}}
    s.routes[("GET", "/sdapi/v1/version")] = lambda r: versions[variant]
    flags = {"a1111": {"api": True}, "forge": {"api": True, "cuda_malloc": False, "always_gpu": False},
             "sdnext": {"use_ipex": False, "backend": "diffusers"}}
    s.routes[("GET", "/sdapi/v1/cmd-flags")] = lambda r: flags[variant]
    s.state_options = {"sd_model_checkpoint": "hassakuSD15_v13.safetensors [7eb674963a]"}
    s.routes[("GET", "/sdapi/v1/options")] = lambda r: s.state_options

    def set_options(r):
        s.state_options.update(r.json)
        return {}

    s.routes[("POST", "/sdapi/v1/options")] = set_options

    def generate(r):
        if s.state.get("reject"):
            return Reply(422, {"detail": [{"loc": ["body", "steps"], "msg": "bad"}]})
        if s.state["interrupted"]:
            return {"images": [], "parameters": r.json, "info": json.dumps({"interrupted": True})}
        n = r.json.get("batch_size", 1)
        seed = r.json["seed"] if r.json.get("seed", -1) >= 0 else 1234
        info = {"seed": seed, "all_seeds": [seed + i for i in range(n)],
                "infotexts": [f"{r.json['prompt']}\nNegative prompt: {r.json.get('negative_prompt','')}\nSteps: {r.json['steps']}, Seed: {seed + i}"
                              for i in range(n)]}
        return {"images": [png_b64() for _ in range(n)], "parameters": r.json, "info": json.dumps(info)}

    s.routes[("POST", "/sdapi/v1/txt2img")] = generate
    s.routes[("POST", "/sdapi/v1/img2img")] = generate

    def interrupt(r):
        s.state["interrupted"] = True
        return {}

    s.routes[("POST", "/sdapi/v1/interrupt")] = interrupt
    s.routes[("GET", "/sdapi/v1/progress")] = lambda r: {"progress": 0.5, "eta_relative": 3.0,
                                                         "state": {"job": "job1", "job_count": 1, "sampling_step": 10, "sampling_steps": 20}}
    return s


# ComfyUI ------------------------------------------------------------------------
COMFY_NODES = {
    "CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [["sd15/hassakuSD15_v13.safetensors", "waiIllustriousSDXL_v170.safetensors"]]}}},
    "UNETLoader": {"input": {"required": {"unet_name": [["flux1-dev.safetensors"]], "weight_dtype": [["default", "fp8_e4m3fn"]]}}},
    "DualCLIPLoader": {"input": {"required": {"clip_name1": [["clip_l.safetensors", "t5xxl_fp16.safetensors"]],
                                              "clip_name2": [["clip_l.safetensors", "t5xxl_fp16.safetensors"]],
                                              "type": [["sdxl", "flux"]]}}},
    "VAELoader": {"input": {"required": {"vae_name": [["ae.safetensors"]]}}},
    "FluxGuidance": {"input": {"required": {"conditioning": ["CONDITIONING"], "guidance": ["FLOAT", {"default": 3.5}]}}},
    "LoraLoader": {"input": {"required": {"model": ["MODEL"], "clip": ["CLIP"], "lora_name": [["detail_tweaker.safetensors", "chars/makima.safetensors"]],
                                          "strength_model": ["FLOAT", {"default": 1.0}], "strength_clip": ["FLOAT", {"default": 1.0}]}}},
    "LoraLoaderModelOnly": {"input": {"required": {"model": ["MODEL"], "lora_name": [["detail_tweaker.safetensors"]],
                                                   "strength_model": ["FLOAT", {"default": 1.0}]}}},
    "KSampler": {"input": {"required": {"model": ["MODEL"], "seed": ["INT", {"default": 0}], "steps": ["INT", {"default": 20}],
                                        "cfg": ["FLOAT", {"default": 8.0}],
                                        "sampler_name": ["COMBO", {"options": ["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde"]}],
                                        "scheduler": [["normal", "karras", "simple", "sgm_uniform"]],
                                        "positive": ["CONDITIONING"], "negative": ["CONDITIONING"], "latent_image": ["LATENT"],
                                        "denoise": ["FLOAT", {"default": 1.0}]}}},
    "CLIPTextEncode": {"input": {"required": {"text": ["STRING", {"multiline": True}], "clip": ["CLIP"]}}},
    "CLIPSetLastLayer": {"input": {"required": {"clip": ["CLIP"], "stop_at_clip_layer": ["INT", {"default": -1}]}}},
    "EmptyLatentImage": {"input": {"required": {"width": ["INT", {"default": 512}], "height": ["INT", {"default": 512}],
                                                "batch_size": ["INT", {"default": 1}]}}},
    "EmptySD3LatentImage": {"input": {"required": {"width": ["INT", {"default": 1024}], "height": ["INT", {"default": 1024}],
                                                   "batch_size": ["INT", {"default": 1}]}}},
    "LoadImage": {"input": {"required": {"image": [["a.png"], {"image_upload": True}]}}},
    "LoadImageMask": {"input": {"required": {"image": [["a.png"]], "channel": [["alpha", "red", "green", "blue"]]}}},
    "VAEEncode": {"input": {"required": {"pixels": ["IMAGE"], "vae": ["VAE"]}}},
    "VAEDecode": {"input": {"required": {"samples": ["LATENT"], "vae": ["VAE"]}}},
    "SetLatentNoiseMask": {"input": {"required": {"samples": ["LATENT"], "mask": ["MASK"]}}},
    "ImageCompositeMasked": {"input": {"required": {"destination": ["IMAGE"], "source": ["IMAGE"], "x": ["INT", {"default": 0}],
                                                    "y": ["INT", {"default": 0}], "resize_source": ["BOOLEAN", {"default": False}]},
                                       "optional": {"mask": ["MASK"]}}},
    "ImageScale": {"input": {"required": {"image": ["IMAGE"], "upscale_method": [["nearest-exact", "bilinear", "lanczos"]],
                                          "width": ["INT"], "height": ["INT"], "crop": [["disabled", "center"]]}}},
    "RepeatLatentBatch": {"input": {"required": {"samples": ["LATENT"], "amount": ["INT", {"default": 1}]}}},
    "MaskToImage": {"input": {"required": {"mask": ["MASK"]}}},
    "ImageToMask": {"input": {"required": {"image": ["IMAGE"], "channel": [["red", "green", "blue", "alpha"]]}}},
    "PreviewImage": {"input": {"required": {"images": ["IMAGE"]}}},
}


def comfyui_server() -> FakeServer:
    s = FakeServer()
    s.state = {"history": {}, "mode": "ok", "queued": []}
    s.routes[("GET", "/system_stats")] = lambda r: {"system": {"comfyui_version": "0.3.99", "python_version": "3.12", "argv": ["main.py"]},
                                                    "devices": [{"name": "mps"}]}

    def object_info(r):
        node = r.path.rsplit("/", 1)[-1]
        return {node: COMFY_NODES[node]} if node in COMFY_NODES else {}

    s.routes[("GET", "/object_info/*")] = object_info
    s.routes[("GET", "/experiment/models")] = lambda r: [{"name": "checkpoints", "folders": ["/comfy/models/checkpoints"]},
                                                         {"name": "loras", "folders": ["/comfy/models/loras"]},
                                                         {"name": "diffusion_models", "folders": ["/comfy/models/diffusion_models"]}]
    s.routes[("GET", "/experiment/models/checkpoints")] = lambda r: [{"name": "sd15/hassakuSD15_v13.safetensors", "pathIndex": 0},
                                                                     {"name": "waiIllustriousSDXL_v170.safetensors", "pathIndex": 0}]
    s.routes[("GET", "/experiment/models/loras")] = lambda r: [{"name": "detail_tweaker.safetensors", "pathIndex": 0}]
    s.routes[("GET", "/experiment/models/diffusion_models")] = lambda r: [{"name": "flux1-dev.safetensors", "pathIndex": 0}]
    s.routes[("POST", "/upload/image")] = lambda r: {"name": "sdmux_upload.png", "subfolder": "", "type": "input"}

    def prompt(r):
        workflow = r.json["prompt"]
        for node_id, node in workflow.items():
            if node["class_type"] not in COMFY_NODES:
                return Reply(400, {"error": {"type": "invalid_prompt", "message": f"Cannot execute because node {node['class_type']} does not exist."},
                                   "node_errors": {}})
            if node["class_type"] == "KSampler" and node["inputs"].get("steps", 1) <= 0:
                return Reply(400, {"error": {"type": "prompt_outputs_failed_validation", "message": "Prompt outputs failed validation"},
                                   "node_errors": {node_id: {"class_type": "KSampler", "errors": [
                                       {"message": "Value 0 smaller than min of 1", "details": "steps"}]}}})
        pid = f"p{len(s.state['queued']) + 1}"
        s.state["queued"].append((pid, workflow))
        output = next(k for k, v in workflow.items() if v["class_type"] in ("PreviewImage", "SaveImage"))
        if s.state["mode"] == "error":
            s.state["history"][pid] = {"status": {"status_str": "error", "completed": False, "messages": [
                ["execution_error", {"node_type": "KSampler", "exception_message": "MPS out of memory"}]]}, "outputs": {}}
        elif s.state["mode"] == "hang":
            pass
        else:
            s.state["history"][pid] = {"status": {"status_str": "success", "completed": True, "messages": []},
                                       "outputs": {output: {"images": [{"filename": "out.png", "subfolder": "", "type": "temp"}]}}}
        return {"prompt_id": pid, "number": 1, "node_errors": {}}

    s.routes[("POST", "/prompt")] = prompt
    s.routes[("GET", "/history/*")] = lambda r: ({r.path.rsplit("/", 1)[-1]: s.state["history"][r.path.rsplit("/", 1)[-1]]}
                                                 if r.path.rsplit("/", 1)[-1] in s.state["history"] else {})
    s.routes[("GET", "/view")] = lambda r: Reply(200, png_bytes(color=(10, 200, 10)), "image/png")

    def interrupt(r):
        for pid, _ in s.state["queued"]:
            if pid not in s.state["history"]:
                s.state["history"][pid] = {"status": {"status_str": "error", "completed": False,
                                                      "messages": [["execution_interrupted", {}]]}, "outputs": {}}
        return {}

    s.routes[("POST", "/interrupt")] = interrupt
    s.routes[("POST", "/queue")] = lambda r: {}
    s.routes[("GET", "/queue")] = lambda r: {"queue_running": [], "queue_pending": []}
    s.routes[("GET", "/view_metadata/checkpoints")] = lambda r: {"modelspec.title": "WAI"}
    return s


# SwarmUI ------------------------------------------------------------------------
def swarmui_server(require_login: bool = False) -> FakeServer:
    s = FakeServer()
    s.state = {"sessions": set(), "next": 0, "logged_in": False, "expire_once": False, "interrupted": False}

    def new_session(r):
        if require_login and not s.state["logged_in"]:
            return {"error_id": "must_login", "error": "You must log in."}
        s.state["next"] += 1
        sid = f"S{s.state['next']}"
        s.state["sessions"].add(sid)
        return {"session_id": sid, "user_id": "local", "version": "0.9.7.1", "server_id": "x"}

    def login(r):
        if r.json.get("username") == "deep" and r.json.get("password") == "pw":
            s.state["logged_in"] = True
            return {"success": True}
        return {"error": "Invalid login."}

    def session_route(fn):
        def wrapped(r):
            if s.state["expire_once"]:
                s.state["expire_once"] = False
                s.state["sessions"].clear()
            if r.json.get("session_id") not in s.state["sessions"]:
                return {"error_id": "invalid_session_id"}
            return fn(r)
        return wrapped

    s.routes[("POST", "/API/GetNewSession")] = new_session
    s.routes[("POST", "/API/Login")] = login
    s.routes[("POST", "/API/ListT2IParams")] = session_route(lambda r: {"list": [
        {"id": "sampler", "values": ["euler///Euler", "euler_ancestral///Euler Ancestral", "dpmpp_2m///DPM++ 2M"]},
        {"id": "scheduler", "values": ["normal///Normal", "karras///Karras", "simple///Simple"]},
    ]})

    def list_models(r):
        if r.json.get("subtype") == "LoRA":
            return {"folders": [], "files": [{"name": "chars/detail.safetensors", "architecture": "stable-diffusion-v1/lora"}]}
        return {"folders": ["sd15"], "files": [
            {"name": "sd15/hassakuSD15_v13.safetensors", "title": "Hassaku", "architecture": "stable-diffusion-v1"},
            {"name": "waiIllustriousSDXL_v170.safetensors", "title": "WAI", "architecture": "stable-diffusion-xl-v1-base"}]}

    s.routes[("POST", "/API/ListModels")] = session_route(list_models)

    def generate(r):
        if s.state["interrupted"]:
            return {"error": "Generation was interrupted."}
        if r.json.get("model") == "broken":
            return {"error": "Model failed to load."}
        n = r.json.get("images", 1)
        seed = r.json["seed"] if r.json.get("seed", -1) >= 0 else 777
        params = json.dumps({"sui_image_params": {"seed": seed, "sampler": r.json.get("sampler"), "prompt": r.json["prompt"]}})
        return {"images": ["data:image/png;base64," + png_b64(text={"parameters": params}) for _ in range(n)]}

    s.routes[("POST", "/API/GenerateText2Image")] = session_route(generate)

    def interrupt(r):
        s.state["interrupted"] = True
        return {"success": True}

    s.routes[("POST", "/API/InterruptAll")] = session_route(interrupt)
    s.routes[("POST", "/API/GetCurrentStatus")] = session_route(lambda r: {"status": {"waiting_gens": 0, "live_gens": 1, "loading_models": 0},
                                                                           "backend_status": {"status": "running", "message": ""}})
    return s


# InvokeAI -----------------------------------------------------------------------
INVOKE_MODELS = {
    "main": [
        {"key": "k1", "hash": "blake3:aa", "name": "hassakuSD15_v13", "base": "sd-1", "type": "main", "path": "/nowhere/hassaku.safetensors"},
        {"key": "k2", "hash": "blake3:bb", "name": "waiIllustriousSDXL_v170", "base": "sdxl", "type": "main", "path": "/nowhere/wai.safetensors"},
        {"key": "k3", "hash": "blake3:cc", "name": "FLUX Schnell", "base": "flux", "type": "main", "path": "/nowhere/flux"},
    ],
    "lora": [{"key": "l1", "hash": "blake3:dd", "name": "detail", "base": "sd-1", "type": "lora"}],
}


def invokeai_server(require_login: bool = False) -> FakeServer:
    s = FakeServer()
    s.state = {"items": {}, "next": 0, "mode": "ok", "graphs": []}

    def authed(fn):
        def wrapped(r):
            if require_login and r.headers.get("Authorization") != "Bearer tok123":
                return Reply(401, {"detail": "Not authenticated"})
            return fn(r)
        return wrapped

    s.routes[("GET", "/api/v1/app/version")] = lambda r: {"version": "6.14.4"}

    def login(r):
        if r.json.get("email") == "a@b.c" and r.json.get("password") == "pw":
            return {"token": "tok123"}
        return Reply(401, {"detail": "bad login"})

    s.routes[("POST", "/api/v1/auth/login")] = login
    s.routes[("GET", "/api/v2/models/")] = authed(lambda r: {"models": INVOKE_MODELS.get(r.query.get("model_type"), [])})
    s.routes[("POST", "/api/v1/images/upload")] = authed(lambda r: {"image_name": f"up{len(s.requests)}.png"})

    def enqueue(r):
        graph = r.json["batch"]["graph"]
        s.state["graphs"].append(graph)
        s.state["next"] += 1
        item = s.state["next"]
        status = {"ok": "completed", "fail": "failed", "hang": "in_progress"}[s.state["mode"]]
        s.state["items"][item] = {
            "item_id": item, "status": status, "error_type": "RuntimeError" if status == "failed" else None,
            "error_message": "boom" if status == "failed" else None,
            "session": {"source_prepared_mapping": {"output": ["prep-out"]},
                        "results": {"prep-out": {"image": {"image_name": "result.png"}}} if status == "completed" else {}},
        }
        return {"item_ids": [item], "batch": {"batch_id": "b"}}

    s.routes[("POST", "/api/v1/queue/default/enqueue_batch")] = authed(enqueue)

    def item(r):
        item_id = int(r.path.rstrip("/").rsplit("/", 1)[-1])
        return s.state["items"][item_id]

    s.routes[("GET", "/api/v1/queue/default/i/*")] = authed(item)

    def cancel(r):
        item_id = int(r.path.split("/i/", 1)[1].split("/")[0])
        s.state["items"][item_id]["status"] = "canceled"
        return s.state["items"][item_id]

    s.routes[("PUT", "/api/v1/queue/default/i/*")] = authed(cancel)
    s.routes[("GET", "/api/v1/images/i/*")] = authed(lambda r: Reply(200, png_bytes(color=(0, 0, 200), size=(64, 64)), "image/png"))
    s.routes[("GET", "/api/v1/queue/default/status")] = authed(lambda r: {"queue": {"in_progress": 1, "pending": 0}})
    return s
