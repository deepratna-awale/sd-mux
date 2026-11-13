import io
import json
import pickle
import struct
import zipfile

import pytest

import sdmux
from sdmux.modelinfo import identify, identify_name
from sdmux.modelinfo.headers import HeaderError, read_header

P = "model.diffusion_model."


def write_st(path, tensors: dict, metadata: dict | None = None):
    header = {}
    offset = 0
    for name, shape in tensors.items():
        header[name] = {"dtype": "F16", "shape": list(shape), "data_offsets": [offset, offset]}
    if metadata is not None:
        header["__metadata__"] = metadata
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
    return str(path)


def unet(width, in_ch=4, extra=None, prefix=P):
    t = {
        prefix + "input_blocks.0.0.weight": (320, in_ch, 3, 3),
        prefix + "input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight": (320, width),
        prefix + "middle_block.1.transformer_blocks.0.attn2.to_k.weight": (1280, width),
        prefix + "out.2.weight": (4, 320, 3, 3),
        "first_stage_model.decoder.conv_in.weight": (512, 4, 3, 3),
    }
    for i in range(6):
        t[f"{prefix}output_blocks.{i}.0.in_layers.0.weight"] = (320,)
    t.update(extra or {})
    return t


def full(arch_keys: dict, prefix=P):
    """A checkpoint-like file: diffusion keys under prefix plus some filler so the prefix wins."""
    t = {prefix + k: v for k, v in arch_keys.items()}
    for i in range(8):
        t[f"{prefix}filler.{i}.weight"] = (4,)
    return t


# Architecture rules -------------------------------------------------------------
@pytest.mark.parametrize(
    "width, arch",
    [(768, "sd15"), (1024, "sd2"), (2048, "sdxl"), (1280, "sdxl-refiner")],
)
def test_unet_context_width(tmp_path, width, arch):
    info = identify(write_st(tmp_path / "m.safetensors", unet(width)))
    assert (info.kind, info.architecture) == ("checkpoint", arch)
    assert info.why("architecture")[0].source == "header"


@pytest.mark.parametrize("channels, variant", [(9, "inpaint"), (8, "instruct"), (4, None)])
def test_unet_variants(tmp_path, channels, variant):
    info = identify(write_st(tmp_path / "m.safetensors", unet(768, channels)))
    assert info.variant == variant


DIT = {
    "sd3": {"joint_blocks.0.context_block.attn.qkv.weight": (4608, 1536)},
    "stable-cascade": {"clf.1.weight": (16, 2048)},
    "auraflow": {"double_layers.0.attn.w1q.weight": (3072, 3072)},
    "hunyuan-dit": {"mlp_t5.0.weight": (8192, 2048)},
    "hunyuan-video": {"txt_in.individual_token_refiner.blocks.0.norm1.weight": (3072,)},
    "flux2": {"double_blocks.0.img_attn.norm.key_norm.scale": (128,), "img_in.weight": (6144, 128),
              "double_stream_modulation_img.lin.weight": (36864, 6144)},
    "chroma": {"double_blocks.0.img_attn.norm.key_norm.scale": (128,), "img_in.weight": (3072, 64),
               "distilled_guidance_layer.norms.0.scale": (5120,)},
    "chroma-radiance": {"double_blocks.0.img_attn.norm.key_norm.scale": (128,),
                        "distilled_guidance_layer.norms.0.scale": (5120,), "nerf_blocks.0.norm.scale": (64,)},
    "mochi": {"t5_yproj.weight": (3072, 4096)},
    "ltxv": {"adaln_single.emb.timestep_embedder.linear_1.bias": (2048,)},
    "pixart": {"t_block.1.weight": (6912, 1152)},
    "cosmos": {"blocks.block0.blocks.0.block.attn.to_q.0.weight": (4096, 4096)},
    "lumina2": {"cap_embedder.1.weight": (2304, 2304), "noise_refiner.0.attention.k_norm.weight": (96,)},
    "cogvideox": {"blocks.0.norm1.linear.weight": (18432, 512)},
    "wan": {"head.modulation": (1, 2, 5120)},
    "hidream": {"caption_projection.0.linear.weight": (2560, 4096)},
    "qwen-image": {"txt_norm.weight": (3584,)},
}


@pytest.mark.parametrize("arch", sorted(DIT))
def test_dit_rules(tmp_path, arch):
    info = identify(write_st(tmp_path / "m.safetensors", full(DIT[arch])))
    assert info.architecture == arch
    assert info.kind == "diffusion_model"


@pytest.mark.parametrize("prefix", ["model.diffusion_model.", "", "model."])
def test_flux_dev_schnell_fill(tmp_path, prefix):
    base = {"double_blocks.0.img_attn.norm.key_norm.scale": (128,), "img_in.weight": (3072, 64)}
    dev = identify(write_st(tmp_path / "dev.safetensors", full({**base, "guidance_in.in_layer.weight": (3072, 256)}, prefix)))
    schnell = identify(write_st(tmp_path / "s.safetensors", full(base, prefix)))
    fill = identify(write_st(tmp_path / "f.safetensors", full({**base, "img_in.weight": (3072, 384)}, prefix)))
    assert (dev.architecture, dev.variant, dev.prediction) == ("flux", "dev", "flow")
    assert (schnell.architecture, schnell.variant) == ("flux", "schnell")
    assert fill.variant == "fill"
    assert schnell.profile.steps == 4


def test_diffusers_flux(tmp_path):
    t = {"transformer_blocks.0.attn.norm_added_k.weight": (128,), "single_transformer_blocks.0.norm.linear.weight": (9216, 3072),
         "x_embedder.weight": (3072, 64), "context_embedder.weight": (3072, 4096)}
    assert identify(write_st(tmp_path / "m.safetensors", t)).architecture == "flux"


@pytest.mark.parametrize(
    "keys, prediction",
    [({"v_pred": (0,)}, "v"), ({"v_pred": (0,), "ztsnr": (0,)}, "v"), ({"edm_mean": (1, 4, 1, 1), "edm_std": (1, 4, 1, 1)}, "edm"),
     ({"edm_vpred.sigma_max": ()}, "v"), ({}, "epsilon")],
)
def test_prediction_markers(tmp_path, keys, prediction):
    info = identify(write_st(tmp_path / "m.safetensors", unet(2048, extra=keys)))
    assert info.prediction == prediction
    assert info.zero_terminal_snr == ("ztsnr" in keys)


def test_vpred_profile(tmp_path):
    info = identify(write_st(tmp_path / "noobaiXLVpred.safetensors", unet(2048, extra={"v_pred": (0,)})))
    assert (info.family, info.prediction) == ("noobai", "v")
    assert info.profile.sampler == "Euler" and info.profile.cfg == 5.0


def test_tolerates_extra_top_level_keys_and_merge_metadata(tmp_path):
    extra = {"alphas_cumprod": (1000,), "model_ema.decay": (), "model_ema.num_updates": (), "betas": (1000,),
             "cond_stage_model.transformer.text_model.final_layer_norm.weight": (768,)}
    meta = {"sd_merge_models": json.dumps({"abc": {"name": "MeinaMix v10.ckpt", "legacy_hash": "x"}}),
            "sd_merge_recipe": json.dumps({"type": "webui"})}
    info = identify(write_st(tmp_path / "merged.safetensors", unet(768, extra=extra), meta))
    assert (info.kind, info.architecture, info.family) == ("checkpoint", "sd15", "anime")
    assert info.why("family")[0].source == "metadata"


# Other kinds ----------------------------------------------------------------------
def test_lora_kinds(tmp_path):
    sdxl = {"lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_down.weight": (8, 768),
            "lora_te2_text_model_encoder_layers_0_mlp_fc1.lora_down.weight": (8, 1280),
            "lora_unet_input_blocks_4_1_transformer_blocks_0_attn2_to_k.lora_down.weight": (8, 2048)}
    info = identify(write_st(tmp_path / "x.safetensors", sdxl, {"ss_sd_model_name": "ponyDiffusionV6XL_v6StartWithThisOne.safetensors",
                                                               "ss_base_model_version": "sdxl_base_v1-0"}))
    assert (info.kind, info.architecture, info.family) == ("lora", "sdxl", "pony")
    assert info.trained_on.startswith("ponyDiffusionV6XL")

    sd15 = {"lora_te_text_model_encoder_layers_0_mlp_fc1.lora_down.weight": (8, 768),
            "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight": (8, 768)}
    info = identify(write_st(tmp_path / "y.safetensors", sd15, {"ss_sd_model_name": "animefull-final-pruned.ckpt"}))
    assert (info.kind, info.architecture, info.family) == ("lora", "sd15", "anime")

    flux = {"transformer.single_transformer_blocks.0.attn.to_k.lora_A.weight": (16, 3072),
            "transformer.single_transformer_blocks.0.attn.to_k.lora_B.weight": (3072, 16)}
    assert identify(write_st(tmp_path / "z.safetensors", flux)).architecture == "flux"


def test_vae_text_encoder_controlnet_embedding(tmp_path):
    vae = {"encoder.down.0.block.0.conv1.weight": (128, 128, 3, 3), "decoder.conv_in.weight": (512, 16, 3, 3)}
    info = identify(write_st(tmp_path / "ae.safetensors", vae))
    assert (info.kind, info.architecture) == ("vae", "flux-vae")
    clip = {"text_model.encoder.layers.0.mlp.fc1.weight": (3072, 768), "text_model.final_layer_norm.weight": (768,)}
    info = identify(write_st(tmp_path / "clip_l.safetensors", clip))
    assert (info.kind, info.text_encoder) == ("text_encoder", "clip-l")
    t5 = {"encoder.block.0.layer.0.SelfAttention.q.weight": (4096, 4096), "shared.weight": (32128, 4096)}
    assert identify(write_st(tmp_path / "t5.safetensors", t5)).text_encoder == "t5"
    cn = {"control_model.input_hint_block.0.weight": (16, 3, 3, 3),
          "control_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight": (320, 768),
          "control_model.input_blocks.0.0.weight": (320, 4, 3, 3)}
    info = identify(write_st(tmp_path / "cn.safetensors", cn))
    assert (info.kind, info.architecture) == ("controlnet", "sd15")
    emb = {"emb_params": (4, 768)}
    assert identify(write_st(tmp_path / "e.safetensors", emb)).kind == "embedding"


def test_unknown_and_not_a_model(tmp_path):
    info = identify(write_st(tmp_path / "x.safetensors", {"foo.weight": (3, 3)}))
    assert info.kind == "unknown" and info.architecture is None
    junk = tmp_path / "junk.safetensors"
    junk.write_bytes(b"not a model at all")
    info = identify(junk)
    assert info.kind == "unknown" and info.conflicts
    with pytest.raises(HeaderError):
        read_header(junk)


# Family sources ---------------------------------------------------------------------
def test_family_by_filename_only(tmp_path):
    info = identify(write_st(tmp_path / "waiIllustriousSDXL_v170.safetensors", unet(2048)))
    assert (info.architecture, info.family, info.prediction) == ("sdxl", "illustrious", "epsilon")
    ev = info.why("family")[0]
    assert (ev.source, ev.confidence) == ("filename", "low")
    p = info.profile
    assert p.dialect == "danbooru" and p.clip_skip == 2 and "masterpiece" in p.quality
    assert p.prompt("1girl, masterpiece") == "best quality, amazing quality, 1girl, masterpiece"


def test_sidecars_beat_filename(tmp_path):
    path = write_st(tmp_path / "ponyish_v1.safetensors", unet(2048))
    (tmp_path / "ponyish_v1.metadata.json").write_text(json.dumps(
        {"sha256": "AB" * 32, "base_model": "Unknown", "civitai": {"baseModel": "Illustrious", "id": 5, "modelId": 6}}))
    info = identify(path)
    assert info.family == "illustrious" and info.why("family")[0].source.startswith("sidecar:")
    assert info.sha256 == "ab" * 32 and info.civitai == {"modelId": 6, "versionId": 5}
    assert any("pony" in c for c in info.conflicts)


@pytest.mark.parametrize(
    "suffix, body, family",
    [
        (".civitai.info", {"baseModel": "NoobAI", "modelId": 1, "id": 2, "files": [{"hashes": {"SHA256": "CD" * 32}}]}, "noobai"),
        (".civitai.json", {"modelId": 1, "versionId": 2, "baseModel": "Pony"}, "pony"),
        (".cm-info.json", {"ModelId": 1, "VersionId": 2, "BaseModel": "Illustrious", "Hashes": {"SHA256": "EF" * 32}}, "illustrious"),
    ],
)
def test_other_sidecar_formats(tmp_path, suffix, body, family):
    path = write_st(tmp_path / "m.safetensors", unet(2048))
    (tmp_path / f"m{suffix}").write_text(json.dumps(body))
    assert identify(path).family == family


def test_civitai_helper_json_sd_version(tmp_path):
    (tmp_path / "x.json").write_text(json.dumps({"sha256": "11" * 32, "modelId": 3, "modelVersionId": 4, "sd version": "SD1"}))
    (tmp_path / "x.safetensors").write_bytes(b"\x00" * 16)  # unreadable header: architecture comes from the sidecar
    info = identify(tmp_path / "x.safetensors")
    assert info.architecture == "sd15" and info.sha256 == "11" * 32


def test_modelspec_metadata(tmp_path):
    meta = {"modelspec.architecture": "stable-diffusion-xl-v1-base", "modelspec.title": "Animagine XL 4.0",
            "modelspec.prediction_type": "epsilon"}
    info = identify(write_st(tmp_path / "m.safetensors", unet(2048), meta))
    assert info.family == "animagine" and info.why("family")[0].source == "metadata"


def test_civitai_online(tmp_path, monkeypatch):
    calls = []

    def fake_lookup(sha, api_key=None, timeout=15.0, hosts=()):
        calls.append(sha)
        return {"id": 2, "modelId": 1, "baseModel": "Pony", "model": {"name": "Some Merge"}, "_host": "https://civitai.red"}

    import sdmux.modelinfo.lookups as lk

    monkeypatch.setattr(lk, "civitai_lookup", fake_lookup)
    monkeypatch.setenv("SDMUX_LOOKUPS", "civitai")
    monkeypatch.setenv("SDMUX_CACHE_DIR", str(tmp_path / "cache"))
    path = write_st(tmp_path / "plain_name.safetensors", unet(2048))
    offline = identify(path)
    assert offline.family is None and not calls
    online = identify(path, online=True)
    assert online.family == "pony" and online.why("family")[0].source == "civitai"
    assert len(calls) == 1 and online.sha256 and len(online.sha256) == 64
    assert online.civitai["site"] == "https://civitai.red"
    assert online.links[0].url == "https://civitai.red/models/1?modelVersionId=2"


def test_identify_name_for_remote_models():
    info = identify_name("ponyDiffusionV6XL.safetensors", architecture="sdxl")
    assert (info.architecture, info.family) == ("sdxl", "pony")
    assert info.profile.quality.startswith("score_9")
    info = identify_name("illustrious_thing.safetensors", architecture="sd15")
    assert info.family is None  # an SDXL family name on an SD1.5 model is dropped


# Other formats -------------------------------------------------------------------------
def test_gguf(tmp_path):
    def s(text):
        b = text.encode()
        return struct.pack("<Q", len(b)) + b

    tensors = [("double_blocks.0.img_attn.norm.key_norm.scale", [128]), ("img_in.weight", [64, 3072]),
               ("guidance_in.in_layer.weight", [256, 3072])]
    out = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(tensors)) + struct.pack("<Q", 1)
    out += s("general.architecture") + struct.pack("<I", 8) + s("flux")
    for name, dims in tensors:
        out += s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<I", 1) + struct.pack("<Q", 0)
    p = tmp_path / "flux1-dev-Q4.gguf"
    p.write_bytes(out)
    h = read_header(p)
    assert h.tensors["img_in.weight"] == (3072, 64)
    info = identify(p)
    assert (info.format, info.architecture, info.variant) == ("gguf", "flux", "dev")


def _torch_like_pickle(keys: dict) -> bytes:
    """A pickle shaped like torch.save output, referencing torch._utils._rebuild_tensor_v2 and os.system.

    Fake modules exist only while dumping; walking the result must neither import nor call anything."""
    import sys
    import types

    torch = types.ModuleType("torch")
    utils = types.ModuleType("torch._utils")

    def _rebuild_tensor_v2(*a):  # pragma: no cover - never called
        raise AssertionError("called")

    _rebuild_tensor_v2.__module__ = "torch._utils"
    _rebuild_tensor_v2.__qualname__ = "_rebuild_tensor_v2"
    utils._rebuild_tensor_v2 = _rebuild_tensor_v2
    torch._utils = utils

    class T:
        def __init__(self, shape):
            self.shape = shape

        def __reduce__(self):
            import os

            return (_rebuild_tensor_v2, ((os.system, ("echo pwned",)), 0, self.shape, (1,) * len(self.shape), False))

    sys.modules["torch"], sys.modules["torch._utils"] = torch, utils
    try:
        return pickle.dumps({"state_dict": {k: T(v) for k, v in keys.items()}, "epoch": 3}, protocol=2)
    finally:
        del sys.modules["torch"], sys.modules["torch._utils"]


def test_ckpt_walked_without_unpickling(tmp_path, capfd):
    keys = {
        "model.diffusion_model.input_blocks.0.0.weight": (320, 4, 3, 3),
        "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight": (320, 1024),
        "cond_stage_model.model.ln_final.weight": (1024,),
    }
    for i in range(6):
        keys[f"model.diffusion_model.f{i}"] = (4,)
    path = tmp_path / "sd21.ckpt"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("archive/data.pkl", _torch_like_pickle(keys))
    h = read_header(path)
    assert h.tensors["model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight"] == (320, 1024)
    info = identify(path)
    assert (info.format, info.architecture) == ("ckpt", "sd2")
    assert "pwned" not in capfd.readouterr().out
    # Legacy (non-zip) torch.save layout: magic, protocol, sys info, the object.
    legacy = tmp_path / "old.pt"
    legacy.write_bytes(pickle.dumps(0x1950A86A20F9469CFC6C, 2) + pickle.dumps(1001, 2) + pickle.dumps({}, 2) + _torch_like_pickle(keys))
    assert identify(legacy).architecture == "sd2"


def test_safetensors_length_starting_with_pickle_byte(tmp_path):
    # A header length whose low byte is 0x80 must not be mistaken for a pickle.
    tensors = {f"lora_unet_x{i}.lora_down.weight": (4, 768) for i in range(3)}
    path = tmp_path / "x.safetensors"
    header = {k: {"dtype": "F16", "shape": list(v), "data_offsets": [0, 0]} for k, v in tensors.items()}
    raw = json.dumps(header).encode()
    raw += b" " * ((0x80 - len(raw)) % 256)
    assert len(raw) % 256 == 0x80
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)
    assert identify(path).kind == "lora"


def test_profile_api():
    from sdmux.modelinfo import build_profile

    p = build_profile("sdxl", "pony")
    assert p.defaults == {"width": 1024, "height": 1024, "steps": 25, "cfg": 7.0, "clip_skip": 2, "sampler": "Euler a"}
    assert p.negative_prompt("lowres") == "lowres, score_4, score_5, score_6"
    assert build_profile("flux", None, "dev").guidance == 3.5
    assert build_profile(None, None).dialect == "clip"


def test_other_lookup_services(tmp_path, monkeypatch):
    import sdmux.modelinfo.lookups as lk

    monkeypatch.setenv("SDMUX_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(lk, "civitai_lookup", lambda *a, **k: None)
    answers = {
        "https://civarchive.com/api/sha256/" + "ab" * 32: (200, {
            "files": [{"filename": "x.safetensors", "source": "huggingface", "url": "https://huggingface.co/a/b/x.safetensors"}],
            "model": {"id": 9, "name": "Some TensorArt Model", "platform": "tensorart", "version": {
                "id": 10, "baseModel": "NoobAI", "platform_url": "https://tensor.art/models/9",
                "mirrors": [{"platform": "seaart", "platform_url": "https://www.seaart.ai/models/detail/q", "name": "mirror"}]}}}),
        "https://modelindex.dev/api/v1/hash/sha256:" + "ab" * 32: (200, {"matches": [
            {"source": "hf", "org": "Laxhar", "name": "noobai-XL-1.1", "path": "NoobAI-XL-v1.1.safetensors"}]}),
        "https://huggingface.co/api/models/Laxhar/noobai-XL-1.1": (200, {
            "cardData": {"base_model": ["Laxhar/noobai-XL-1.0"]}, "tags": ["diffusers:StableDiffusionXLPipeline"]}),
    }
    seen = []

    def fake_get(url, timeout, headers=None):
        seen.append(url)
        return answers.get(url, (404, {"errors": []}))

    monkeypatch.setattr(lk, "_get_json", fake_get)
    path = write_st(tmp_path / "plain.safetensors", unet(2048))
    (tmp_path / "plain.json").write_text(json.dumps({"sha256": "AB" * 32, "sd version": "SDXL"}))
    info = identify(path, online=True)
    assert info.family == "noobai" and info.why("family")[0].source == "civarchive:tensorart"
    services = {l.service for l in info.links}
    assert {"tensorart", "seaart", "huggingface"} <= services
    results = {r.service: r for r in lk.lookup_all("ab" * 32)}
    assert results["huggingface"].facts[0].value == "sdxl"
    assert any(f.value == "noobai" for f in results["modelindex"].facts)
    # Answers are cached: asking again makes no new requests.
    count = len(seen)
    lk.lookup_all("ab" * 32)
    assert len(seen) == count
    # An archive answer without a "model" (stray files sharing the hash) counts as not found.
    answers["https://civarchive.com/api/sha256/" + "cd" * 32] = (200, {"files": [{"filename": "last.pt", "source": "huggingface"}]})
    assert not lk.lookup_civarchive("cd" * 32).found


def test_not_found_answers_expire(tmp_path, monkeypatch):
    import sdmux.modelinfo.civitai as cv
    import sdmux.modelinfo.lookups as lk

    monkeypatch.setenv("SDMUX_CACHE_DIR", str(tmp_path / "cache"))
    seen = []
    monkeypatch.setattr(lk, "_get_json", lambda url, timeout, headers=None: (seen.append(url), (404, None))[1])
    sha = "ef" * 32
    assert not lk.lookup_modelindex(sha).found and not lk.lookup_civarchive(sha).found
    assert len(seen) == 2
    lk.lookup_modelindex(sha), lk.lookup_civarchive(sha)
    assert len(seen) == 2  # a fresh "not found" is trusted
    monkeypatch.setenv("SDMUX_NEGATIVE_TTL", "0")
    monkeypatch.setattr(cv.time, "time", lambda: 1e12)
    lk.lookup_modelindex(sha), lk.lookup_civarchive(sha)
    assert len(seen) == 4  # an expired one is asked again

    class Response:
        def __init__(self, status, body=None):
            self.status_code, self._body = status, body

        def json(self):
            return self._body

    import requests

    answers = {"n": 0}

    def fake_get(url, headers=None, timeout=None):
        answers["n"] += 1
        return Response(404) if answers["n"] <= 2 else Response(200, {"id": 1, "modelId": 2, "baseModel": "Pony"})

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.delenv("SDMUX_NEGATIVE_TTL")
    assert cv.civitai_lookup(sha) is None and cv.civitai_lookup(sha) is None
    assert answers["n"] == 2  # both hosts asked once, then the miss is cached
    monkeypatch.setattr(cv.time, "time", lambda: 1e12 + cv.NEGATIVE_TTL + 1)
    assert cv.civitai_lookup(sha)["baseModel"] == "Pony"
