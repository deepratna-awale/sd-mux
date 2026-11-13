import socket
import threading
import time

import pytest
from PIL import Image

import sdmux
from sdmux.backends import A1111, ComfyUI, DrawThings, Forge, InvokeAI, SDCpp, SDNext, SwarmUI

from .fakes import FakeServer, Reply, a1111_server, comfyui_server, drawthings_server, invokeai_server, sdcpp_server, swarmui_server


def img(color=(255, 255, 255), size=(64, 64)):
    return Image.new("RGB", size, color)


def mask(size=(64, 64)):
    m = Image.new("L", size, 0)
    m.paste(255, (16, 16, 48, 48))
    return m


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# Detection ------------------------------------------------------------------------
@pytest.mark.parametrize(
    "factory, kind",
    [
        (lambda: a1111_server("a1111"), "a1111"),
        (lambda: a1111_server("forge"), "forge"),
        (lambda: a1111_server("forge2"), "forge"),
        (lambda: a1111_server("sdnext"), "sdnext"),
        (comfyui_server, "comfyui"),
        (swarmui_server, "swarmui"),
        (invokeai_server, "invokeai"),
    ],
)
def test_detect(factory, kind):
    with factory() as s:
        assert sdmux.detect(s.url).kind == kind
        sd = sdmux.connect(s.url)
        assert sd.kind == kind


def test_detect_swarm_requiring_login_still_detected():
    with swarmui_server(require_login=True) as s:
        assert sdmux.detect(s.url).kind == "swarmui"


def test_detect_unreachable():
    with pytest.raises(sdmux.Unreachable):
        sdmux.detect(f"http://127.0.0.1:{free_port()}", connect_timeout=1)


def test_detect_not_a_backend():
    with FakeServer() as s:
        with pytest.raises(sdmux.NotReady) as e:
            sdmux.detect(s.url)
        assert "GET /system_stats" in str(e.value)


def test_detect_auth_refused():
    with a1111_server() as s:
        s.require_auth = ("deep", "pw")
        with pytest.raises(sdmux.AuthRefused):
            sdmux.detect(s.url)
        assert sdmux.detect(s.url, username="deep", password="pw").kind == "a1111"


def test_subpath_and_url_credentials():
    with a1111_server(prefix="/proxy/7860") as s:
        s.require_auth = ("deep", "p@ss")
        url = s.url.replace("http://", "http://deep:p%40ss@")
        sd = sdmux.connect(url)
        assert sd.kind == "a1111"
        assert "deep" not in sd.url
        assert [m.name for m in sd.models()][0] == "hassakuSD15_v13"


def test_unknown_kind():
    with pytest.raises(ValueError):
        sdmux.connect("http://127.0.0.1:1", kind="nope")


def test_timeout_is_typed():
    with a1111_server() as s:
        s.routes[("GET", "/sdapi/v1/sd-models")] = lambda r: Reply(200, [], delay=1.5)
        sd = A1111(s.url, timeout=0.3, retries=0)
        with pytest.raises(sdmux.TimedOut):
            sd.models()


# A1111 family ---------------------------------------------------------------------
@pytest.mark.parametrize("cls", [A1111, Forge, SDNext])
def test_a1111_listings(cls):
    with a1111_server("sdnext" if cls is SDNext else "a1111") as s:
        sd = cls(s.url)
        assert [m.name for m in sd.models()] == ["hassakuSD15_v13", "sdxl_waiIllustriousSDXL_v170"]
        assert sd.models()[0].path == "/nowhere/hassakuSD15_v13.safetensors"
        assert [m.name for m in sd.loras()] == ["detail"]
        assert "Euler a" in sd.samplers()
        assert sd.schedulers() == (list(SDNext.SIGMAS) if cls is SDNext else ["Automatic", "Karras", "Normal"])
        assert sd.info()["version"]


def test_a1111_txt2img_payload_and_result():
    with a1111_server() as s:
        sd = A1111(s.url)
        r = sd.txt2img("1girl, smile <lora:Detail:0.6>", negative="lowres", model="waiIllustrious", width=640, height=512,
                       steps=12, cfg=6, sampler="euler_ancestral", scheduler="karras", seed=42, clip_skip=2)
        body = s.last("POST", "/sdapi/v1/txt2img").json
        assert body["prompt"] == "1girl, smile <lora:detail:0.6>"
        assert body["sampler_name"] == "Euler a"
        assert body["scheduler"] == "Karras"
        assert body["override_settings"] == {"sd_model_checkpoint": "sdxl/waiIllustriousSDXL_v170.safetensors [abc]",
                                             "CLIP_stop_at_last_layers": 2}
        assert body["override_settings_restore_afterwards"] is False
        assert (body["width"], body["height"], body["steps"], body["cfg_scale"], body["seed"]) == (640, 512, 12, 6, 42)
        assert r.seed == 42 and r.seeds == [42]
        assert isinstance(r.image, Image.Image)
        assert r.image.info["parameters"].startswith("1girl")
        assert r.parameters == r.image.info["parameters"]


def test_a1111_combined_sampler_name_on_old_server():
    with a1111_server("sdnext") as s:  # no /schedulers route, like A1111 before 1.10
        sd = A1111(s.url)
        sd.txt2img("x", sampler="DPM++ 2M", scheduler="Karras")
        body = s.last("POST", "/sdapi/v1/txt2img").json
        assert body["sampler_name"] == "DPM++ 2M Karras"
        assert "scheduler" not in body


def test_sdnext_scheduler_as_sigma_setting():
    with a1111_server("sdnext") as s:
        sd = SDNext(s.url)
        sd.txt2img("x", sampler="DPM++ 2M Karras", clip_skip=2)
        body = s.last("POST", "/sdapi/v1/txt2img").json
        assert body["sampler_name"] == "DPM++ 2M"
        assert body["override_settings"]["schedulers_sigma"] == "karras"
        assert body["clip_skip"] == 2
        sd.txt2img("x", model="waiIllustrious")
        assert s.state_options["sd_model_checkpoint"] == "sdxl/waiIllustriousSDXL_v170.safetensors [abc]"
        assert "sd_model_checkpoint" not in (s.last("POST", "/sdapi/v1/txt2img").json.get("override_settings") or {})


def test_a1111_img2img_and_inpaint():
    with a1111_server() as s:
        sd = A1111(s.url)
        r = sd.img2img(img(size=(96, 64)), "a cat", denoise=0.4, seed=5)
        body = s.last("POST", "/sdapi/v1/img2img").json
        assert body["denoising_strength"] == 0.4 and len(body["init_images"]) == 1
        assert (body["width"], body["height"]) == (96, 64)
        assert "mask" not in body
        r = sd.inpaint(img(), mask(), "a dog", denoise=0.7, batch_size=2)
        body = s.last("POST", "/sdapi/v1/img2img").json
        assert body["mask"] and body["inpainting_mask_invert"] == 0 and body["inpainting_fill"] == 1
        assert len(r.images) == 2 and r.seeds == [1234, 1235]


def test_a1111_extra_passthrough_and_raw_api():
    with a1111_server() as s:
        sd = A1111(s.url)
        sd.txt2img("x", extra={"enable_hr": True, "hr_scale": 1.5})
        assert s.last("POST", "/sdapi/v1/txt2img").json["enable_hr"] is True
        assert sd.api("sd-models", method="GET")[0]["model_name"] == "hassakuSD15_v13"


def test_a1111_errors():
    with a1111_server() as s:
        sd = A1111(s.url)
        with pytest.raises(sdmux.Rejected):
            sd.txt2img("x", model="no-such-model")
        with pytest.raises(sdmux.Rejected):
            sd.txt2img("x <lora:missing:1>")
        s.state["reject"] = True
        with pytest.raises(sdmux.Rejected) as e:
            sd.txt2img("x")
        assert "422" in str(e.value)


def test_a1111_progress_and_interrupt():
    with a1111_server() as s:
        sd = A1111(s.url)
        p = sd.progress()
        assert p.active and p.step == 10 and p.steps == 20 and p.fraction == 0.5
        sd.interrupt()
        with pytest.raises(sdmux.Cancelled):
            sd.txt2img("x")


# ComfyUI ---------------------------------------------------------------------------
def test_comfy_listings():
    with comfyui_server() as s:
        sd = ComfyUI(s.url)
        models = sd.models()
        assert [m.name for m in models] == ["sd15/hassakuSD15_v13.safetensors", "waiIllustriousSDXL_v170.safetensors",
                                            "flux1-dev.safetensors"]
        assert models[0].path == "/comfy/models/checkpoints/sd15/hassakuSD15_v13.safetensors"
        assert models[2].kind == "diffusion_model"
        assert sd.samplers() == ["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde"]
        assert "karras" in sd.schedulers()
        assert sd.info()["version"] == "0.3.99"
        assert sd.ready()["ready"]


def test_comfy_txt2img_graph():
    with comfyui_server() as s:
        sd = ComfyUI(s.url)
        r = sd.txt2img("1girl <lora:detail_tweaker:0.6>", negative="lowres", model="hassaku", steps=10, cfg=5,
                       sampler="Euler a", scheduler="Karras", seed=42, clip_skip=2, width=512, height=768)
        wf = s.last("POST", "/prompt").json["prompt"]
        assert wf["ckpt"]["inputs"]["ckpt_name"] == "sd15/hassakuSD15_v13.safetensors"
        assert wf["lora_0"]["inputs"]["lora_name"] == "detail_tweaker.safetensors"
        assert wf["lora_0"]["inputs"]["strength_model"] == 0.6
        assert wf["clip_skip"]["inputs"]["stop_at_clip_layer"] == -2
        assert wf["positive"]["inputs"]["text"] == "1girl"
        ks = wf["sampler"]["inputs"]
        assert (ks["sampler_name"], ks["scheduler"], ks["seed"], ks["steps"], ks["denoise"]) == ("euler_ancestral", "karras", 42, 10, 1.0)
        assert wf["latent"]["inputs"] == {"width": 512, "height": 768, "batch_size": 1}
        assert r.seed == 42 and "Sampler: Euler a" in r.parameters and "<lora:detail_tweaker:0.6>" in r.parameters


def test_comfy_combined_sampler_and_unknown_sampler():
    with comfyui_server() as s:
        sd = ComfyUI(s.url)
        sd.txt2img("x", sampler="DPM++ 2M Karras")
        ks = s.last("POST", "/prompt").json["prompt"]["sampler"]["inputs"]
        assert (ks["sampler_name"], ks["scheduler"]) == ("dpmpp_2m", "karras")
        with pytest.raises(sdmux.Rejected):
            sd.txt2img("x", sampler="Nonexistent Sampler")


def test_comfy_img2img_and_inpaint_graph():
    with comfyui_server() as s:
        sd = ComfyUI(s.url)
        sd.img2img(img(size=(64, 64)), "a cat", denoise=0.4, width=128, height=128)
        wf = s.last("POST", "/prompt").json["prompt"]
        assert wf["scaled"]["inputs"]["width"] == 128
        assert wf["encode"]["inputs"]["pixels"] == ["scaled", 0]
        assert wf["sampler"]["inputs"]["denoise"] == 0.4
        sd.inpaint(img(), mask(), "a dog", denoise=0.7, batch_size=2)
        wf = s.last("POST", "/prompt").json["prompt"]
        assert wf["mask"]["class_type"] == "LoadImageMask" and wf["mask"]["inputs"]["channel"] == "red"
        assert wf["noise_mask"]["inputs"]["mask"] == ["mask", 0]
        assert wf["batch"]["inputs"]["amount"] == 2
        assert wf["composite"]["inputs"]["mask"] == ["mask", 0]
        assert wf["output"]["inputs"]["images"] == ["composite", 0]


def test_comfy_split_file_flux_graph():
    with comfyui_server() as s:
        sd = ComfyUI(s.url)
        flux = next(m for m in sd.models() if m.kind == "diffusion_model")
        flux.architecture = "flux"
        sd.txt2img("a photo of a fox", model=flux, steps=4, cfg=1, seed=1)
        wf = s.last("POST", "/prompt").json["prompt"]
        assert wf["unet"]["inputs"]["unet_name"] == "flux1-dev.safetensors"
        assert wf["clip"]["inputs"] == {"clip_name1": "clip_l.safetensors", "clip_name2": "t5xxl_fp16.safetensors", "type": "flux"}
        assert wf["guidance"]["inputs"]["guidance"] == 3.5
        assert wf["latent"]["class_type"] == "EmptySD3LatentImage"


def test_comfy_split_file_unknown_arch_unsupported():
    with comfyui_server() as s:
        sd = ComfyUI(s.url)
        sd.models = lambda: [sdmux.Model(name="mystery.safetensors", kind="diffusion_model")]
        with pytest.raises(sdmux.Unsupported):
            sd.txt2img("x", model="mystery")
        # A diffusion model whose file is not on this machine is identified by name.
        del sd.models
        sd.txt2img("x", model="flux1-dev")
        assert s.last("POST", "/prompt").json["prompt"]["clip"]["inputs"]["type"] == "flux"


def test_comfy_rejection_and_execution_error():
    with comfyui_server() as s:
        sd = ComfyUI(s.url)
        with pytest.raises(sdmux.Rejected) as e:
            sd.txt2img("x", steps=0)
        assert "KSampler: Value 0 smaller than min of 1 steps" in str(e.value)
        s.state["mode"] = "error"
        with pytest.raises(sdmux.Rejected) as e:
            sd.txt2img("x")
        assert "MPS out of memory" in str(e.value)


def test_comfy_run_workflow_escape_hatch():
    with comfyui_server() as s:
        sd = ComfyUI(s.url)
        wf = {"1": {"class_type": "PreviewImage", "inputs": {}}}
        r = sd.run_workflow(wf)
        assert len(r.images) == 1 and r.raw["workflow"] is wf


def test_comfy_interrupt_and_timeout():
    with comfyui_server() as s:
        s.state["mode"] = "hang"
        sd = ComfyUI(s.url, poll_interval=0.05)
        errors = []

        def run():
            try:
                sd.txt2img("x")
            except sdmux.SdmuxError as e:
                errors.append(e)

        t = threading.Thread(target=run)
        t.start()
        time.sleep(0.4)
        sd.interrupt()
        t.join(5)
        assert errors and isinstance(errors[0], sdmux.Cancelled)
        assert s.last("POST", "/interrupt").json == {"prompt_id": "p1"}  # the running prompt, nothing else
        sd2 = ComfyUI(s.url, poll_interval=0.05, job_timeout=0.3)
        with pytest.raises(sdmux.TimedOut):
            sd2.txt2img("x")
        # The timed-out job is interrupted on the server, not left running.
        assert s.last("POST", "/interrupt").json == {"prompt_id": "p2"}
        assert s.state["history"]["p2"]["status"]["messages"][0][0] == "execution_interrupted"


def test_comfy_interrupt_before_prompt_id_is_known():
    with comfyui_server() as s:
        s.state["mode"] = "hang"
        sd = ComfyUI(s.url, poll_interval=0.05)
        real_queue = sd.queue

        def queue_then_interrupt(workflow):
            prompt_id = real_queue(workflow)
            sd.interrupt()  # arrives before the client has recorded the prompt id
            return prompt_id

        sd.queue = queue_then_interrupt
        with pytest.raises(sdmux.Cancelled):
            sd.txt2img("x")
        interrupts = [r.json for r in s.requests if (r.method, r.path) == ("POST", "/interrupt")]
        assert interrupts == [{"prompt_id": "p1"}]  # never a bare /interrupt that could stop another client's job


def test_interrupt_not_cleared_by_waiting_call():
    with a1111_server() as s:
        sd = A1111(s.url)
        entered, release, seen = threading.Event(), threading.Event(), []

        def running_job():
            with sd._job():
                entered.set()
                release.wait(5)
                seen.append(sd._cancel.is_set())

        first = threading.Thread(target=running_job)
        first.start()
        entered.wait(5)
        waiting = threading.Thread(target=lambda: sd.txt2img("x"))
        waiting.start()
        time.sleep(0.3)
        sd._cancel.set()  # interrupt() aimed at the running job
        release.set()
        first.join(5)
        waiting.join(5)
        assert seen == [True]


# SwarmUI ---------------------------------------------------------------------------
def test_swarm_listings_and_generation():
    with swarmui_server() as s:
        sd = SwarmUI(s.url)
        models = sd.models()
        assert [(m.name, m.architecture) for m in models] == [("sd15/hassakuSD15_v13.safetensors", "sd15"),
                                                              ("waiIllustriousSDXL_v170.safetensors", "sdxl")]
        assert sd.samplers() == ["euler", "euler_ancestral", "dpmpp_2m"]
        r = sd.txt2img("1girl <lora:detail:0.5>", model="hassaku", sampler="Euler a", scheduler="Karras", seed=9, clip_skip=2,
                       steps=10, width=512, height=512)
        body = s.last("POST", "/API/GenerateText2Image").json
        assert body["prompt"] == "1girl <lora:chars/detail:0.5>"
        assert (body["sampler"], body["scheduler"], body["clipstopatlayer"], body["seed"]) == ("euler_ancestral", "karras", -2, 9)
        assert body["model"] == "sd15/hassakuSD15_v13.safetensors"
        assert r.seed == 9 and "Seed: 9" in r.parameters


def test_swarm_img2img_inpaint_and_session_renewal():
    with swarmui_server() as s:
        sd = SwarmUI(s.url)
        sd.img2img(img(), "x", denoise=0.3)
        body = s.last("POST", "/API/GenerateText2Image").json
        assert body["initimagecreativity"] == 0.3 and body["initimage"] and "maskimage" not in body
        s.state["expire_once"] = True
        sd.inpaint(img(), mask(), "y", denoise=0.8)
        body = s.last("POST", "/API/GenerateText2Image").json
        assert body["maskimage"] and body["initimagecreativity"] == 0.8


def test_swarm_login_and_errors():
    with swarmui_server(require_login=True) as s:
        with pytest.raises(sdmux.AuthRefused):
            SwarmUI(s.url).models()
        with pytest.raises(sdmux.AuthRefused):
            SwarmUI(s.url, username="deep", password="nope").models()
        sd = SwarmUI(s.url, username="deep", password="pw")
        assert sd.models()
        with pytest.raises(sdmux.Rejected):
            sd.txt2img("x", model="no-such")
        sd.models = lambda: [sdmux.Model(name="broken")]
        with pytest.raises(sdmux.Rejected) as e:
            sd.txt2img("x", model="broken")
        assert "failed to load" in str(e.value)


def test_swarm_progress_and_interrupt():
    with swarmui_server() as s:
        sd = SwarmUI(s.url)
        assert sd.progress().active
        sd.interrupt()
        with pytest.raises(sdmux.Cancelled):
            sd.txt2img("x")


# InvokeAI --------------------------------------------------------------------------
def test_invoke_listings():
    with invokeai_server() as s:
        sd = InvokeAI(s.url)
        assert [(m.name, m.architecture) for m in sd.models()] == [("hassakuSD15_v13", "sd15"), ("waiIllustriousSDXL_v170", "sdxl"),
                                                                   ("FLUX Schnell", "flux")]
        assert [m.name for m in sd.loras()] == ["detail"]
        assert "euler_a" in sd.samplers()
        assert sd.ready()["ready"]


def test_invoke_txt2img_graph_sd15_with_lora():
    with invokeai_server() as s:
        sd = InvokeAI(s.url)
        r = sd.txt2img("1girl <lora:detail:0.7>", model="hassaku", sampler="DPM++ 2M", scheduler="Karras", seed=3, clip_skip=2,
                       steps=9, width=515, height=512)
        g = s.state["graphs"][-1]
        n = g["nodes"]
        assert n["model_loader"]["type"] == "main_model_loader"
        assert n["clip_skip"]["skipped_layers"] == 1
        assert n["lora_0"]["weight"] == 0.7
        assert n["denoise"]["scheduler"] == "dpmpp_2m_k"
        assert n["denoise"]["denoising_start"] == 0.0
        assert n["noise"]["width"] == 512 and n["noise"]["seed"] == 3
        assert n["positive"]["prompt"] == "1girl"
        assert r.seed == 3


def test_invoke_sdxl_img2img_and_inpaint():
    with invokeai_server() as s:
        sd = InvokeAI(s.url)
        sd.img2img(img(), "x", model="waiIllustrious", denoise=0.25)
        n = s.state["graphs"][-1]["nodes"]
        assert n["model_loader"]["type"] == "sdxl_model_loader"
        assert n["positive"]["type"] == "sdxl_compel_prompt"
        assert n["denoise"]["denoising_start"] == 0.75
        assert "i2l" in n and "gradient_mask" not in n
        r = sd.inpaint(img(), mask(), "y", model="waiIllustrious", denoise=1.0)
        n = s.state["graphs"][-1]["nodes"]
        assert n["gradient_mask"]["type"] == "create_gradient_mask"
        edges = s.state["graphs"][-1]["edges"]
        assert {"source": {"node_id": "gradient_mask", "field": "denoise_mask"},
                "destination": {"node_id": "denoise", "field": "denoise_mask"}} in edges
        # Outside the mask the original pixels are kept (the fake returns solid blue).
        assert r.image.getpixel((0, 0)) == (255, 255, 255)
        assert r.image.getpixel((32, 32)) == (0, 0, 200)


def test_invoke_unsupported_base_and_failure_and_cancel():
    with invokeai_server() as s:
        sd = InvokeAI(s.url, poll_interval=0.05)
        with pytest.raises(sdmux.Unsupported):
            sd.txt2img("x", model="FLUX Schnell")
        s.state["mode"] = "fail"
        with pytest.raises(sdmux.Rejected) as e:
            sd.txt2img("x")
        assert "boom" in str(e.value)
        s.state["mode"] = "hang"
        errors = []
        t = threading.Thread(target=lambda: errors.append(pytest.raises(sdmux.Cancelled, sd.txt2img, "x")))
        t.start()
        time.sleep(0.4)
        sd.interrupt()
        t.join(5)
        assert errors


def test_invoke_login():
    with invokeai_server(require_login=True) as s:
        with pytest.raises(sdmux.AuthRefused):
            InvokeAI(s.url).models()
        with pytest.raises(sdmux.AuthRefused):
            InvokeAI(s.url, username="a@b.c", password="no").models()
        assert InvokeAI(s.url, username="a@b.c", password="pw").models()
        assert InvokeAI(s.url, token="tok123").models()


# Transport details carried over from AutoExpress ---------------------------------------
def test_extra_headers_secrets_and_redaction():
    from sdmux.transport import Options, Transport, parse_headers, redact_url

    assert parse_headers("CF-Access-Client-Id: abc\nCF-Access-Client-Secret: xyz") == {
        "CF-Access-Client-Id": "abc", "CF-Access-Client-Secret": "xyz"}
    assert parse_headers('{"X-Key": "1"}') == {"X-Key": "1"}
    for bad in ("Host: evil", "no colon here", {"X-A": "a\r\nInjected: 1"}, {"bad name": "x"}):
        with pytest.raises(ValueError):
            parse_headers(bad)
    assert redact_url("https://deep:secret@pod-7860.proxy.runpod.net/x") == "https://***@pod-7860.proxy.runpod.net/x"
    opts = Options(username="deep", password="secret", token="tok", headers={"X-Key": "k"})
    assert "secret" not in repr(opts) and "tok" not in repr(opts)
    t = Transport("http://127.0.0.1:1", Options(username="deep", password="pw", headers="X-Key: k"))
    h = t.auth_headers()
    assert h["Authorization"].startswith("Basic ") and h["X-Key"] == "k"


def test_extra_headers_are_sent():
    with a1111_server() as s:
        A1111(s.url, headers={"CF-Access-Client-Id": "abc"}).models()
        assert s.last("GET", "/sdapi/v1/sd-models").headers.get("CF-Access-Client-Id") == "abc"


def test_comfy_view_retries_a_dropped_download():
    with comfyui_server() as s:
        calls = {"n": 0}
        ok = s.routes[("GET", "/view")]

        def flaky(r):
            calls["n"] += 1
            return Reply(502, {"error": "proxy"}) if calls["n"] == 1 else ok(r)

        s.routes[("GET", "/view")] = flaky
        r = ComfyUI(s.url).txt2img("x")
        assert r.image and calls["n"] == 2


# Defaults from the model profile -------------------------------------------------------
def test_profile_defaults_fill_missing_settings():
    with comfyui_server() as s:
        sd = ComfyUI(s.url)
        r = sd.txt2img("1girl", model="waiIllustrious", seed=1)
        wf = s.last("POST", "/prompt").json["prompt"]
        ks = wf["sampler"]["inputs"]
        assert wf["latent"]["inputs"]["width"] == 1024 and (ks["steps"], ks["cfg"]) == (28, 6.0)
        assert ks["sampler_name"] == "euler_ancestral"
        assert wf["clip_skip"]["inputs"]["stop_at_clip_layer"] == -2
        assert r.defaults["steps"] == (28, "profile:waiIllustriousSDXL_v170.safetensors")
        # Anything passed explicitly wins; a sampler without a scheduler is not mixed with the profile's.
        r = sd.txt2img("1girl", model="waiIllustrious", steps=5, sampler="DPM++ 2M", width=768, height=768)
        ks = s.last("POST", "/prompt").json["prompt"]["sampler"]["inputs"]
        assert (ks["steps"], ks["sampler_name"], ks["scheduler"]) == (5, "dpmpp_2m", "normal")
        assert "steps" not in r.defaults and "scheduler" not in r.defaults
        # img2img keeps the input image's size.
        sd.img2img(img(size=(96, 64)), "x", model="waiIllustrious")
        assert "scaled" not in s.last("POST", "/prompt").json["prompt"]


def test_fixed_defaults_and_current_model():
    with comfyui_server() as s:
        sd = ComfyUI(s.url, defaults="fixed")
        r = sd.txt2img("1girl", model="waiIllustrious")
        wf = s.last("POST", "/prompt").json["prompt"]
        assert wf["latent"]["inputs"]["width"] == 512 and wf["sampler"]["inputs"]["steps"] == 20
        assert "clip_skip" not in wf and r.defaults["steps"] == (20, "fallback")
    with pytest.raises(ValueError):
        ComfyUI("http://127.0.0.1:1", defaults="nope")
    with a1111_server() as s:
        A1111(s.url).txt2img("x")  # no model=: the profile of the loaded checkpoint (hassaku: SD 1.5 anime)
        body = s.last("POST", "/sdapi/v1/txt2img").json
        assert (body["width"], body["steps"], body["sampler_name"], body["scheduler"]) == (512, 25, "DPM++ 2M", "Karras")
        assert body["override_settings"]["CLIP_stop_at_last_layers"] == 2


def test_profile_lookup_retried_after_transient_error():
    with comfyui_server() as s:
        sd = ComfyUI(s.url)
        real, calls = sd.identify, {"n": 0}

        def flaky(model, online=False):
            calls["n"] += 1
            if calls["n"] == 1:
                raise sdmux.TimedOut("slow")
            return real(model, online=online)

        sd.identify = flaky
        assert sd.txt2img("1girl", model="waiIllustrious").defaults["steps"] == (20, "fallback")
        r = sd.txt2img("1girl", model="waiIllustrious")
        assert r.defaults["steps"] == (28, "profile:waiIllustriousSDXL_v170.safetensors")


# ComfyUI node contract and graph builder --------------------------------------------------
def test_comfy_node_discovery():
    with comfyui_server() as s:
        sd = ComfyUI(s.url)
        names = [n.name for n in sd.nodes()]
        assert "FaceDetailer" in names and names == sorted(names)
        fd = sd.node("FaceDetailer")
        assert fd.pack == "ComfyUI-Impact-Pack" and sd.node("KSampler").pack == "core"
        assert fd.inputs["guide_size"].min == 64 and not fd.inputs["wildcard"].required
        assert [o.name for o in fd.outputs] == ["image", "cropped_refined", "mask"]
        ks = sd.node("KSampler")
        assert ks.inputs["sampler_name"].options[:2] == ["euler", "euler_ancestral"]
        assert "out 0 LATENT" in str(ks)
        assert sd.node("NoSuchNode") is None


def test_comfy_graph_builder_runs_any_node():
    with comfyui_server() as s:
        sd = ComfyUI(s.url)
        g = sd.graph()
        ckpt = g.add("CheckpointLoaderSimple", ckpt_name="waiIllustrious")      # loose file-name match
        pos = g.add("CLIPTextEncode", text="1girl", clip=ckpt.CLIP)
        neg = g.add("CLIPTextEncode", text="lowres", clip=ckpt.clip)            # case-insensitive output names
        lat = g.add("EmptyLatentImage", width=1024, height=1024)
        smp = g.add("KSampler", model=ckpt.MODEL, positive=pos, negative=neg, latent_image=lat, steps=10)
        img = g.add("VAEDecode", samples=smp, vae=ckpt[2])
        up = g.add("ImageUpscaleWithModel", upscale_model=g.add("UpscaleModelLoader", model_name="4x-AnimeSharp"), image=img)
        fd = g.add("FaceDetailer", image=up, model=ckpt.MODEL)
        g.add("PreviewImage", images=fd.image)
        api = g.to_api()
        assert api["1"]["inputs"]["ckpt_name"] == "waiIllustriousSDXL_v170.safetensors"
        assert api["5"]["inputs"]["sampler_name"] == "euler" and api["5"]["inputs"]["cfg"] == 8.0   # server defaults
        assert api["5"]["inputs"]["model"] == ["1", 0] and api["6"]["inputs"]["vae"] == ["1", 2]
        r = sd.run(g)
        assert r.images and set(r.raw["outputs"]) == {"10"}
        assert s.last("POST", "/prompt").json["prompt"] == api


def test_comfy_graph_validation_reports_every_problem():
    with comfyui_server() as s:
        sd = ComfyUI(s.url)
        g = sd.graph()
        ckpt = g.add("CheckpointLoaderSimple", ckpt_name="nope.safetensors")
        g.add("KSampler", model=ckpt.CLIP, steps=0, sampler_name="bogus", typo=1)
        g.add("NotInstalled", x=1)
        with pytest.raises(sdmux.GraphError) as e:
            sd.run(g)
        text = "\n".join(e.value.problems)
        for expected in ("'nope.safetensors' is not one of the options", "expects MODEL, got CLIP", "0 is below the minimum 1",
                         "'bogus' is not one of the options", "no input 'typo'", "missing required input 'positive'",
                         "no node 'NotInstalled'", "no output node"):
            assert expected in text, expected
        assert not any(r.method == "POST" and r.path == "/prompt" for r in s.requests)
        with pytest.raises(ValueError):
            g.add("KSampler", model=ckpt)            # three outputs: must pick one
        with pytest.raises(AttributeError):
            _ = ckpt.LATENT


def test_comfy_patch_hook_and_workflow():
    with comfyui_server() as s:
        sd = ComfyUI(s.url)
        g = sd.workflow(prompt="1girl", model="hassaku", steps=4, seed=1)
        assert g.ref("sampler").inputs["steps"] == 4 and g.find("KSampler")

        def add_upscale(graph):
            loader = graph.add("UpscaleModelLoader", model_name="RealESRGAN_x4plus")
            up = graph.add("ImageUpscaleWithModel", upscale_model=loader, image=graph.ref("decode"))
            graph.ref("output").set(images=up)

        sd.txt2img("1girl", model="hassaku", steps=4, patch=add_upscale)
        wf = s.last("POST", "/prompt").json["prompt"]
        up_id = next(k for k, v in wf.items() if v["class_type"] == "ImageUpscaleWithModel")
        assert wf["output"]["inputs"]["images"] == [up_id, 0]
        with pytest.raises(sdmux.GraphError):
            sd.txt2img("1girl", model="hassaku", patch=lambda g: g.ref("sampler").set(steps=-1))
    with a1111_server() as s:
        with pytest.raises(sdmux.Unsupported):
            A1111(s.url).txt2img("x", patch=lambda g: None)


# stable-diffusion.cpp ---------------------------------------------------------------
def test_sdcpp_detection_and_generation():
    with sdcpp_server() as s:
        sd = sdmux.connect(s.url)
        assert isinstance(sd, SDCpp) and sdmux.detect(s.url).kind == "sdcpp"
        assert [m.name for m in sd.models()] == ["sd_v1.5"] and sd.loras()[0].path == "loras/detail_tweaker.safetensors"
        r = sd.txt2img("a cat <lora:detail_tweaker:0.5>", model="sd_v1.5", sampler="Euler a", scheduler="Karras",
                       clip_skip=2, seed=3, steps=8)
        body = s.last("POST", "/sdapi/v1/txt2img").json
        assert body["prompt"] == "a cat" and "<lora" not in body["prompt"]
        assert body["lora"] == [{"path": "loras/detail_tweaker.safetensors", "multiplier": 0.5}]
        assert (body["sampler_name"], body["scheduler"], body["clip_skip"]) == ("euler_a", "karras", 2)
        assert "override_settings" not in body and r.image
        sd.img2img(img(), "a dog", denoise=0.4)
        assert s.last("POST", "/sdapi/v1/img2img").json["denoising_strength"] == 0.4
        with pytest.raises(sdmux.Rejected):
            sd.txt2img("x", model="flux1-dev")  # only the loaded model is listed
        with pytest.raises(sdmux.Unsupported):
            sd.interrupt()


# Draw Things ------------------------------------------------------------------------
def test_drawthings_detection_and_generation():
    with drawthings_server() as s:
        sd = sdmux.connect(s.url)
        assert isinstance(sd, DrawThings) and sdmux.detect(s.url).kind == "drawthings"
        assert [m.name for m in sd.models()] == ["sd_v1.5_f16.ckpt"]
        assert sd.loras()[0].name == "detail_tweaker_lora_f16.ckpt"
        r = sd.txt2img("a cat <lora:detail_tweaker:0.8>", negative="blurry", width=520, height=770, steps=12, cfg=5,
                       sampler="DPM++ 2M", scheduler="Karras", seed=9, clip_skip=2)
        body = s.last("POST", "/sdapi/v1/txt2img").json
        assert body["prompt"] == "a cat" and body["loras"] == [{"file": "detail_tweaker_lora_f16.ckpt", "weight": 0.8}]
        assert (body["width"], body["height"]) == (512, 768)  # multiples of 64
        assert (body["guidance_scale"], body["sampler"], body["seed"], body["clip_skip"]) == (5, "DPM++ 2M Karras", 9, 2)
        assert "strength" not in body and r.seed == 9 and "Seed: 9" in r.parameters
        sd.txt2img("x", sampler="Euler a", scheduler="Align Your Steps", model="other_model_f16.ckpt")
        body = s.last("POST", "/sdapi/v1/txt2img").json
        assert (body["sampler"], body["model"]) == ("Euler A AYS", "other_model_f16.ckpt")
        sd.img2img(img(size=(100, 60)), "a dog", denoise=0.3, width=512, height=512)
        body = s.last("POST", "/sdapi/v1/img2img").json
        assert body["strength"] == 0.3 and len(body["init_images"]) == 1
        with pytest.raises(sdmux.Rejected):
            sd.txt2img("x", sampler="Heun")
        with pytest.raises(sdmux.Unsupported):
            sd.inpaint(img(), mask(), "x")
        with pytest.raises(sdmux.Unsupported):
            sd.interrupt()
