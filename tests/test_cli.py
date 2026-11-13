import json
import socket

from PIL import Image

from sdmux.cli import main

from .fakes import a1111_server, comfyui_server
from .test_identify import unet, write_st


def run(capsys, *argv):
    code = main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


def test_detect(capsys):
    with a1111_server() as s:
        code, out, _ = run(capsys, "detect", s.url)
        assert code == 0 and out.startswith("a1111")
        code, out, _ = run(capsys, "detect", s.url, "--json")
        assert code == 0 and json.loads(out)["kind"] == "a1111"


def test_models_and_loras(capsys):
    with a1111_server() as s:
        code, out, _ = run(capsys, "models", s.url)
        assert code == 0 and "hassakuSD15_v13" in out and "samplers:" in out
        code, out, _ = run(capsys, "models", s.url, "--loras", "--json")
        assert code == 0 and isinstance(json.loads(out), list)


def test_nodes(capsys):
    with comfyui_server() as s:
        code, out, _ = run(capsys, "nodes", s.url, "--pack", "ComfyUI-Impact-Pack")
        assert code == 0 and "FaceDetailer" in out and "KSampler" not in out
        code, out, _ = run(capsys, "nodes", s.url, "KSampler", "--json")
        assert code == 0 and "input" in json.loads(out)
        code, _, err = run(capsys, "nodes", s.url, "NotInstalled")
        assert code == 1 and "NotInstalled" in err
    with a1111_server() as s:
        code, _, err = run(capsys, "nodes", s.url)
        assert code == 2 and "ComfyUI" in err


def test_identify(capsys, tmp_path):
    write_st(tmp_path / "ponyDiffusionV6XL.safetensors", unet(2048))
    (tmp_path / "notes.txt").write_text("not a model")
    code, out, _ = run(capsys, "identify", str(tmp_path))
    assert code == 0 and "ponyDiffusionV6XL.safetensors: checkpoint / sdxl / pony" in out and "notes" not in out
    code, out, _ = run(capsys, "identify", str(tmp_path / "ponyDiffusionV6XL.safetensors"), "--json")
    assert code == 0 and json.loads(out)[0]["architecture"] == "sdxl"
    code, _, err = run(capsys, "identify", str(tmp_path / "missing.safetensors"))
    assert code == 1 and "missing.safetensors" in err


def test_txt2img(capsys, tmp_path):
    out_file = tmp_path / "out.png"
    with a1111_server() as s:
        code, out, _ = run(capsys, "txt2img", s.url, "a red apple", "-o", str(out_file), "--steps", "5", "--seed", "7")
        assert code == 0 and "seed" in out
        body = s.last("POST", "/sdapi/v1/txt2img").json
        assert (body["prompt"], body["steps"], body["seed"]) == ("a red apple", 5, 7)
    assert Image.open(out_file).size[0] > 0


def test_errors_exit_2(capsys):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    code, _, err = run(capsys, "detect", f"http://127.0.0.1:{port}")
    assert code == 2 and err.startswith("error (")
