"""
Live verification against a running backend. Not part of the offline suite.

    python tests/live_check.py http://127.0.0.1:7860 --model hassakuSD15_v13 [--out DIR] [--size 512] [--steps 8]

Exercises detection, listings, txt2img, img2img, inpaint, progress and interrupt, and prints one line per check.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time

from PIL import Image, ImageDraw

import sdmux


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--model")
    ap.add_argument("--lora")
    ap.add_argument("--out", default=".")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--sampler", default="Euler")
    ap.add_argument("--scheduler", default="Karras")
    ap.add_argument("--username")
    ap.add_argument("--password")
    ap.add_argument("--skip", default="", help="comma list of checks to skip")
    args = ap.parse_args()
    auth = {k: v for k, v in (("username", args.username), ("password", args.password)) if v}
    os.makedirs(args.out, exist_ok=True)
    skip = set(filter(None, args.skip.split(",")))
    report: dict = {}

    def check(name, fn):
        if name in skip:
            print(f"SKIP {name}")
            return None
        started = time.monotonic()
        try:
            value = fn()
            report[name] = {"ok": True, "seconds": round(time.monotonic() - started, 1), "value": value}
            print(f"OK   {name} ({time.monotonic() - started:.1f}s): {value}")
            return value
        except Exception as e:  # report and continue
            report[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            print(f"FAIL {name}: {type(e).__name__}: {e}")
            return None

    d = check("detect", lambda: sdmux.detect(args.url, **auth).__dict__)
    sd = sdmux.connect(args.url, **auth)
    check("info", sd.info)
    models = check("models", lambda: [m.name for m in sd.models()])
    check("loras", lambda: [m.name for m in sd.loras()][:10])
    check("samplers", sd.samplers)
    check("schedulers", sd.schedulers)
    check("identify(model)", lambda: (lambda i: {"architecture": i.architecture, "family": i.family, "prediction": i.prediction,
                                                 "via": [str(e) for e in i.evidence if e.field in ("architecture", "family")]})(
        sd.identify(args.model)))
    check("profile(model)", lambda: sd.profile(args.model).defaults)
    size = args.size
    common = dict(model=args.model, steps=args.steps, cfg=6, sampler=args.sampler, scheduler=args.scheduler, width=size,
                  height=size, clip_skip=2)
    prompt = "1girl, smile, upper body, simple background" + (f" <lora:{args.lora}:0.6>" if args.lora else "")

    def t2i():
        r = sd.txt2img(prompt, negative="lowres, bad anatomy", seed=42, **common)
        r.save(os.path.join(args.out, f"{sd.kind}_txt2img.png"))
        return {"seed": r.seed, "size": r.image.size, "parameters": r.parameters.splitlines()[-1][:200]}

    check("txt2img", t2i)
    base = Image.open(os.path.join(args.out, f"{sd.kind}_txt2img.png")) if os.path.exists(os.path.join(args.out, f"{sd.kind}_txt2img.png")) else Image.new("RGB", (size, size), (180, 160, 140))

    def i2i():
        r = sd.img2img(base, "1girl, smile, upper body, red background", denoise=0.5, seed=7, **{k: v for k, v in common.items()})
        r.save(os.path.join(args.out, f"{sd.kind}_img2img.png"))
        return {"seed": r.seed, "size": r.image.size}

    check("img2img", i2i)

    def inpaint():
        mask = Image.new("L", base.size, 0)
        ImageDraw.Draw(mask).rectangle((0, 0, base.size[0] - 1, base.size[1] // 3), fill=255)
        r = sd.inpaint(base, mask, "1girl, wearing a big red hat", denoise=0.9, seed=9, **common)
        r.save(os.path.join(args.out, f"{sd.kind}_inpaint.png"))
        out = r.image.convert("RGB").resize(base.size)
        # Pixels far outside the mask should be close to the input.
        a, b = base.convert("RGB").getpixel((size // 2, size - 4)), out.getpixel((size // 2, size - 4))
        changed_top = sum(abs(x - y) for x, y in zip(base.convert("RGB").getpixel((size // 2, 8)), out.getpixel((size // 2, 8))))
        return {"seed": r.seed, "bottom_pixel_delta": sum(abs(x - y) for x, y in zip(a, b)), "top_pixel_delta": changed_top}

    check("inpaint", inpaint)

    def interrupt():
        outcome = {}

        def run():
            try:
                sd.txt2img(prompt, seed=1, **{**common, "steps": 60})
                outcome["result"] = "finished (not interrupted)"
            except sdmux.Cancelled as e:
                outcome["result"] = f"Cancelled: {e}"
            except Exception as e:
                outcome["result"] = f"{type(e).__name__}: {e}"

        t = threading.Thread(target=run)
        t.start()
        progress = None
        for _ in range(60):
            time.sleep(1)
            try:
                p = sd.progress()
            except Exception as e:
                progress = f"progress failed: {e}"
                break
            if p.active and ((p.step or 0) >= 2 or (p.step is None and _ > 8)):
                progress = p.__dict__
                break
        sd.interrupt()
        t.join(180)
        return {"progress_seen": progress, **outcome}

    check("interrupt", interrupt)
    with open(os.path.join(args.out, f"{sd.kind}_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    return 0 if all(v["ok"] for v in report.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
