"""
AUTOMATIC1111, Forge / reForge and SD.Next, through the /sdapi/v1 API (A1111 and Forge need --api).

Native escape hatch: api(path, payload) posts any body to any /sdapi/v1 route, and
generate() merges Request.extra into the payload, so every field the WebUI accepts
(alwayson_scripts, hires-fix, refiner, ...) is reachable.
"""

from __future__ import annotations

import json
import time
from typing import Any

from ..errors import AuthRefused, Cancelled, Rejected, SdmuxError, TimedOut, Unreachable
from ..names import canonical_scheduler, extract_loras, lora_tags, pick_native, sampler_id
from ..types import IMG2IMG, INPAINT, TXT2IMG, Model, Progress, Request, Result, load_image, to_b64
from .base import Backend


class A1111(Backend):
    kind = "a1111"
    label = "AUTOMATIC1111"
    default_url = "http://127.0.0.1:7860"

    #: When False (sdmux's choice), a model chosen with model= stays loaded after the call, like picking it
    #: in the UI. A1111's own default (True) switches back after every call, reloading weights twice.
    restore_settings = False

    def __init__(self, url=None, options=None, **kw) -> None:
        super().__init__(url, options, **kw)
        self._schedulers: list[str] | None = None

    # Listings -------------------------------------------------------------
    def models(self) -> list[Model]:
        return [
            Model(
                name=m.get("model_name") or m.get("title"),
                title=m.get("title"),
                path=m.get("filename"),
                hash=m.get("hash"),
                sha256=m.get("sha256"),
                raw=m,
            )
            for m in self.http.get_json("/sdapi/v1/sd-models")
        ]

    def loras(self) -> list[Model]:
        response = self.http.request("GET", "/sdapi/v1/loras")
        if response.status_code == 404:
            return []
        items = self.http.json(self.http.check(response, "GET /sdapi/v1/loras"))
        loras = []
        for lora in items:
            metadata = lora.get("metadata") or {}
            loras.append(
                Model(
                    name=lora.get("name") or lora.get("alias"),
                    title=lora.get("alias"),
                    path=lora.get("path"),
                    sha256=metadata.get("sshs_model_hash") if isinstance(metadata, dict) else None,
                    kind="lora",
                    raw=lora,
                )
            )
        return loras

    def samplers(self) -> list[str]:
        return [s["name"] for s in self.http.get_json("/sdapi/v1/samplers")]

    def schedulers(self) -> list[str]:
        """Scheduler labels. [] where there is no /schedulers route (A1111 before 1.10, SD.Next)."""
        if self._schedulers is None:
            response = self.http.request("GET", "/sdapi/v1/schedulers")
            if response.status_code >= 400:
                self._schedulers = []  # 404 before A1111 1.10
            else:
                items = self.http.json(self.http.check(response, "GET /sdapi/v1/schedulers"))
                self._schedulers = [s.get("label") or s.get("name") for s in items]
        return list(self._schedulers)

    def info(self) -> dict:
        info = {"backend": self.kind, "url": self.url}
        response = self.http.request("GET", "/sdapi/v1/version", timeout=10)
        if response.status_code == 200:
            try:
                info["version"] = response.json()
            except ValueError:
                pass
        try:
            options = self.http.get_json("/sdapi/v1/options", timeout=10)
            info["model"] = options.get("sd_model_checkpoint")
        except Rejected:
            pass
        return info

    def current_model(self) -> str | None:
        """The checkpoint loaded on the server, which a request without model= runs on."""
        try:
            return self.http.get_json("/sdapi/v1/options", timeout=30).get("sd_model_checkpoint") or None
        except (Unreachable, TimedOut, AuthRefused):
            raise
        except SdmuxError:
            return None

    # Native ---------------------------------------------------------------
    def api(self, path: str, payload: dict | None = None, method: str = "POST", timeout: float | None = None):
        """Call any /sdapi/v1 route (path may omit the prefix) and return its JSON."""
        if not path.startswith("/"):
            path = "/sdapi/v1/" + path
        response = self.http.request(method, path, json=payload, timeout=timeout)
        return self.http.json(self.http.check(response, f"{method} {path}"))

    # Generation -----------------------------------------------------------
    def _sampler_fields(self, request: Request) -> dict:
        from ..names import canonical_sampler

        fields = {}
        folded = None
        if request.sampler:
            _, folded = canonical_sampler(request.sampler)
            native = pick_native(request.sampler, self.samplers(), sampler_id)
            fields["sampler_name"] = native or request.sampler
        scheduler = request.scheduler or folded
        if scheduler:
            schedulers = self.schedulers()
            if schedulers:
                fields["scheduler"] = pick_native(scheduler, schedulers, canonical_scheduler) or scheduler
            elif request.sampler:
                # No scheduler field (old A1111): A1111 reads a scheduler glued to the sampler name.
                if canonical_scheduler(scheduler) not in ("automatic", "normal") and "sampler_name" in fields:
                    from ..names import a1111_scheduler_label

                    fields["sampler_name"] = f"{fields['sampler_name']} {a1111_scheduler_label(scheduler)}"
        return fields

    def build_payload(self, request: Request) -> dict:
        """The /sdapi/v1/txt2img or /img2img body for a request (LoRA tags stay in the prompt)."""
        prompt, loras = extract_loras(request.prompt)
        if loras:
            available = self.loras()
            _, matched = self.resolve_loras(request.prompt, available)
            prompt = " ".join(p for p in (prompt, lora_tags([(self.lora_name(m), w) for m, w in matched])) if p)
        overrides: dict = {}
        if request.model:
            overrides["sd_model_checkpoint"] = self.resolve_model(request.model).title or request.model
        if request.clip_skip:
            overrides["CLIP_stop_at_last_layers"] = int(request.clip_skip)
        if request.vae:
            overrides["sd_vae"] = request.vae
        payload: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": request.negative,
            "width": request.width,
            "height": request.height,
            "steps": request.steps,
            "cfg_scale": request.cfg,
            "seed": request.seed if request.seed is not None else -1,
            "batch_size": request.batch_size,
            "n_iter": 1,
            "send_images": True,
            "save_images": False,
        }
        sampler_fields = self._sampler_fields(request)
        overrides.update(sampler_fields.pop("override_settings", {}))
        payload.update(sampler_fields)
        if overrides:
            payload["override_settings"] = overrides
            payload["override_settings_restore_afterwards"] = self.restore_settings
        if request.mode in (IMG2IMG, INPAINT):
            payload["init_images"] = [to_b64(request.image.convert("RGB"))]
            payload["denoising_strength"] = request.denoise
            payload["resize_mode"] = 0
        if request.mode == INPAINT:
            payload.update(
                {
                    "mask": to_b64(request.mask.convert("L")),
                    "mask_blur": request.mask_blur,
                    "inpainting_fill": 1,  # start from the original pixels under the mask
                    "inpaint_full_res": request.inpaint_full_res,
                    "inpaint_full_res_padding": request.inpaint_padding,
                    "inpainting_mask_invert": 0,  # white = repaint
                }
            )
        payload.update(request.extra or {})
        return payload

    def _generate(self, request: Request) -> Result:
        payload = self.build_payload(request)
        route = "/sdapi/v1/txt2img" if request.mode == TXT2IMG else "/sdapi/v1/img2img"
        with self._job():
            response = self.http.request("POST", route, json=payload, timeout=self.options.job_timeout)
            cancelled = self._cancel.is_set()
        if response.status_code >= 400:
            raise Rejected(f"{self.label} rejected the request ({response.status_code}): {_error_text(response)}",
                           backend=self.kind, details=response.text)
        body = self.http.json(response)
        images_b64 = body.get("images") or []
        try:
            info = json.loads(body.get("info") or "{}")
        except (TypeError, ValueError):
            info = {}
        if not images_b64:
            if cancelled or info.get("interrupted"):
                raise Cancelled("Interrupted.", backend=self.kind)
            raise Rejected(f"{self.label} returned no image.", backend=self.kind, details=body)
        infotexts = info.get("infotexts") or []
        seeds = info.get("all_seeds") or ([info["seed"]] if "seed" in info else [])
        images = []
        # Extensions may append extra images (masks, grids) beyond the batch; keep only the batch.
        for index, data in enumerate(images_b64[: max(request.batch_size, 1)]):
            image = load_image(data)
            if index < len(infotexts):
                image.info["parameters"] = infotexts[index]
            images.append(image)
        if cancelled:
            # A1111 answers an interrupted job with the partially denoised image; it is kept in details.
            raise Cancelled("Interrupted.", backend=self.kind, details={"partial_images": images, "info": info})
        return Result(
            images=images,
            seed=seeds[0] if seeds else info.get("seed"),
            seeds=list(seeds),
            parameters=infotexts[0] if infotexts else "",
            backend=self.kind,
            raw={"info": info, "parameters": body.get("parameters")},
        )

    # Control --------------------------------------------------------------
    def progress(self) -> Progress:
        r = self.http.get_json("/sdapi/v1/progress", params={"skip_current_image": "true"}, timeout=10)
        state = r.get("state") or {}
        active = bool(state.get("job")) or bool(state.get("job_count"))
        step, steps = state.get("sampling_step") or 0, state.get("sampling_steps") or 0
        eta = r.get("eta_relative")
        return Progress(
            fraction=float(r.get("progress") or 0.0) if active else 0.0,
            step=step if active else None,
            steps=steps if active else None,
            eta=round(eta, 1) if active and eta and eta > 0 else None,
            active=active,
        )

    def interrupt(self) -> None:
        self._cancel.set()
        self.http.check(self.http.request("POST", "/sdapi/v1/interrupt", timeout=10), "interrupt")


def _error_text(response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:1000]
    if isinstance(body, dict):
        return str(body.get("errors") or body.get("detail") or body.get("error") or body)[:1000]
    return str(body)[:1000]


class Forge(A1111):
    """Stable Diffusion WebUI Forge and reForge: the A1111 API with different built-in scripts."""

    kind = "forge"
    label = "Forge"


class SDNext(A1111):
    """
    SD.Next: the same /sdapi/v1 routes; it serves the API without --api.

    SD.Next runs diffusers schedulers: the sampler list holds them all, and the A1111 scheduler is
    approximated by its "Sigma algorithm" setting (schedulers_sigma: default, karras, exponential,
    betas, lambdas, flowmatch). Its /sdapi/v1/schedulers lists samplers, not schedulers.
    """

    kind = "sdnext"
    label = "SD.Next"
    SIGMAS = ("default", "karras", "betas", "exponential", "lambdas", "flowmatch")

    def schedulers(self) -> list[str]:
        return list(self.SIGMAS)

    def _sampler_fields(self, request: Request) -> dict:
        from ..names import canonical_sampler

        fields: dict = {}
        folded = None
        if request.sampler:
            _, folded = canonical_sampler(request.sampler)
            fields["sampler_name"] = pick_native(request.sampler, self.samplers(), sampler_id) or request.sampler
        scheduler = request.scheduler or folded
        if scheduler:
            cid = canonical_scheduler(scheduler)
            sigma = cid if cid in self.SIGMAS else ("betas" if cid == "beta" else "default")
            fields["override_settings"] = {"schedulers_sigma": sigma}
        return fields

    def build_payload(self, request: Request) -> dict:
        payload = super().build_payload(request)
        if request.clip_skip:
            payload["clip_skip"] = int(request.clip_skip)
        # SD.Next ignores sd_model_checkpoint in override_settings when no model is loaded yet
        # ("Aborted: op=process model not loaded"), so the model is switched through /options instead.
        overrides = payload.get("override_settings") or {}
        wanted = overrides.pop("sd_model_checkpoint", None)
        if wanted:
            current = self.http.get_json("/sdapi/v1/options", timeout=30).get("sd_model_checkpoint") or ""
            if wanted not in (current, current.split(" [")[0]):
                self.http.check(self.http.request("POST", "/sdapi/v1/options", json={"sd_model_checkpoint": wanted},
                                                  timeout=self.options.job_timeout), "model switch")
        if not overrides:
            payload.pop("override_settings", None)
            payload.pop("override_settings_restore_afterwards", None)
        return payload
