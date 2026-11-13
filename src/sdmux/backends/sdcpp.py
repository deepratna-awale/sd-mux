"""
stable-diffusion.cpp's server (sd-server) through its /sdapi/v1 routes.

It serves txt2img, img2img (with a mask), progress, and the model, LoRA, sampler and scheduler listings like A1111,
with these differences:

- LoRAs are not read from the prompt; they go in a "lora" list of {path, multiplier}, path as /sdapi/v1/loras lists it.
- One model is loaded at a time (/sdapi/v1/sd-models lists only it); switching is done with /sdcpp/v1/models/load.
- clip_skip is a top-level field; there are no override_settings.
- There is no interrupt route.
"""

from __future__ import annotations

from ..errors import Unsupported
from ..names import extract_loras
from ..types import Model, Request
from .a1111 import A1111


class SDCpp(A1111):
    kind = "sdcpp"
    label = "stable-diffusion.cpp"
    default_url = "http://127.0.0.1:1234"

    def models(self) -> list[Model]:
        return [Model(name=m.get("model_name") or m.get("title"), title=m.get("title"), path=m.get("filename"), raw=m)
                for m in self.http.get_json("/sdapi/v1/sd-models")]

    def loras(self) -> list[Model]:
        return [Model(name=item.get("name") or item.get("path"), title=item.get("display_path"), path=item.get("path"),
                      kind="lora", raw=item)
                for item in self.http.get_json("/sdapi/v1/loras")]

    def local_path(self, model: Model) -> str | None:
        return None  # paths here are relative to the server's model folders

    def info(self) -> dict:
        info = {"backend": self.kind, "url": self.url, "model": self.current_model()}
        response = self.http.request("GET", "/sdcpp/v1/capabilities", timeout=10)
        if response.status_code == 200:
            try:
                info["capabilities"] = response.json()
            except ValueError:
                pass
        return info

    def build_payload(self, request: Request) -> dict:
        if request.vae:
            raise Unsupported("stable-diffusion.cpp picks the VAE when the model is loaded; vae= cannot change it.",
                              backend=self.kind)
        lora_free = Request(**{**request.__dict__, "prompt": extract_loras(request.prompt)[0], "model": None, "clip_skip": None})
        payload = super().build_payload(lora_free)
        payload.pop("override_settings", None)
        payload.pop("override_settings_restore_afterwards", None)
        if request.model:
            self.resolve_model(request.model)  # only the loaded model is listed: anything else is Rejected
        if request.clip_skip:
            payload["clip_skip"] = int(request.clip_skip)
        _, loras = self.resolve_loras(request.prompt)
        if loras:
            payload["lora"] = [{"path": m.path or m.name, "multiplier": weight} for m, weight in loras]
        if "hr_second_pass_steps" in payload:
            steps = payload.pop("hr_second_pass_steps")
            if steps:
                payload["hr_steps"] = steps
        return payload

    def upscale(self, image, upscaler: str | None = None, scale: float = 2.0):
        raise Unsupported("stable-diffusion.cpp upscales with a fixed-scale ESRGAN model through its own API; "
                          "use sd.api('/sdcpp/v1/upscale', ...).", backend=self.kind)

    def interrupt(self) -> None:
        raise Unsupported("stable-diffusion.cpp's /sdapi/v1 API has no interrupt.", backend=self.kind)
