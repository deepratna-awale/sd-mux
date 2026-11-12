"""
sdmux command line.

    sdmux detect <url>
    sdmux models <url> [--loras] [--json]
    sdmux identify <file-or-folder>... [--online] [--json] [--recursive]
    sdmux txt2img <url> "<prompt>" [-o out.png] [--model M] [--steps N] ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys

MODEL_EXTENSIONS = (".safetensors", ".ckpt", ".pt", ".pth", ".gguf", ".sft", ".bin")


def _auth(args) -> dict:
    return {k: v for k, v in {"username": args.username, "password": args.password, "token": args.token}.items() if v}


def cmd_detect(args) -> int:
    import sdmux

    d = sdmux.detect(args.url, **_auth(args))
    if args.json:
        print(json.dumps(d.__dict__, indent=2))
    else:
        print(f"{d.kind}  ({d.evidence}{', version ' + str(d.version) if d.version else ''})")
    return 0


def cmd_models(args) -> int:
    import sdmux

    sd = sdmux.connect(args.url, **_auth(args))
    items = sd.loras() if args.loras else sd.models()
    if args.json:
        print(json.dumps([{"name": m.name, "title": m.title, "path": m.path, "kind": m.kind, "architecture": m.architecture,
                           "sha256": m.sha256} for m in items], indent=2))
        return 0
    print(f"{sd.kind} at {sd.url}: {len(items)} {'LoRAs' if args.loras else 'models'}")
    for m in items:
        extra = f"  [{m.architecture}]" if m.architecture else ""
        print(f"  {m.name}{extra}")
    if not args.loras:
        print(f"samplers: {', '.join(sd.samplers())}")
        schedulers = sd.schedulers()
        if schedulers:
            print(f"schedulers: {', '.join(schedulers)}")
    return 0


def _files(paths: list[str], recursive: bool) -> list[str]:
    out = []
    for path in paths:
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                dirs.sort()
                for name in sorted(files):
                    if name.lower().endswith(MODEL_EXTENSIONS):
                        out.append(os.path.join(root, name))
                if not recursive:
                    break
        else:
            out.append(path)
    return out


def cmd_identify(args) -> int:
    from .modelinfo import identify

    results = []
    status = 0
    for path in _files(args.paths, args.recursive):
        try:
            sources = tuple(x.strip() for x in args.sources.split(",")) if args.sources else None
            info = identify(path, online=args.online or bool(args.sources), sources=sources)
        except OSError as e:
            print(f"{path}: {e}", file=sys.stderr)
            status = 1
            continue
        results.append(info)
        if not args.json:
            _print_info(info, args.verbose)
    if args.json:
        print(json.dumps([i.to_dict(metadata=args.verbose) for i in results], indent=2, default=str))
    return status


def _print_info(info, verbose: bool) -> None:
    fields = [info.kind, info.architecture or "?"]
    if info.family:
        fields.append(info.family)
    if info.variant:
        fields.append(info.variant)
    if info.prediction:
        fields.append(info.prediction)
    print(f"{info.name}: {' / '.join(fields)}")
    for e in info.evidence:
        if e.field in ("kind", "architecture", "family", "variant", "prediction") or verbose:
            print(f"    {e.field:<12} {str(e.value):<14} {e.confidence:<6} {e.source}: {e.detail}")
    for c in info.conflicts:
        print(f"    note: {c}")
    if info.trained_on:
        print(f"    trained on: {info.trained_on}")
    for link in info.links[:6]:
        print(f"    on {link.service}: {link.url}")
    if info.kind in ("checkpoint", "diffusion_model"):
        p = info.profile
        print(f"    profile: {p.dialect}, {p.width}x{p.height}, {p.steps} steps, cfg {p.cfg:g}, {p.sampler}"
              f"{' / ' + p.scheduler if p.scheduler else ''}, clip skip {p.clip_skip}"
              + (f", quality '{p.quality}'" if p.quality else ""))


def cmd_txt2img(args) -> int:
    import sdmux

    sd = sdmux.connect(args.url, **_auth(args))
    kwargs = {k: v for k, v in {"model": args.model, "steps": args.steps, "cfg": args.cfg, "sampler": args.sampler,
                                "scheduler": args.scheduler, "seed": args.seed, "width": args.width, "height": args.height,
                                "clip_skip": args.clip_skip}.items() if v is not None}
    result = sd.txt2img(args.prompt, negative=args.negative or "", **kwargs)
    result.save(args.output)
    print(f"{args.output}  seed {result.seed}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sdmux", description="One client for every local image generation backend.")
    sub = parser.add_subparsers(dest="command", required=True)

    def remote(p):
        p.add_argument("url")
        p.add_argument("--username")
        p.add_argument("--password")
        p.add_argument("--token")
        p.add_argument("--json", action="store_true")

    p = sub.add_parser("detect", help="which backend answers at a URL")
    remote(p)
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser("models", help="list a backend's models (or --loras)")
    remote(p)
    p.add_argument("--loras", action="store_true")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("identify", help="identify model files from their headers")
    p.add_argument("paths", nargs="+")
    p.add_argument("--online", action="store_true",
                   help="also look the sha256 up online: Civitai (.com and .red), CivArchive, modelindex.dev, Hugging Face")
    p.add_argument("--sources", help="comma list of online services to ask (civitai,civarchive,modelindex,huggingface)")
    p.add_argument("--json", action="store_true")
    p.add_argument("-r", "--recursive", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_identify)

    p = sub.add_parser("txt2img", help="generate one image")
    remote(p)
    p.add_argument("prompt")
    p.add_argument("-n", "--negative")
    p.add_argument("-o", "--output", default="sdmux.png")
    p.add_argument("--model")
    p.add_argument("--steps", type=int)
    p.add_argument("--cfg", type=float)
    p.add_argument("--sampler")
    p.add_argument("--scheduler")
    p.add_argument("--seed", type=int)
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--clip-skip", type=int)
    p.set_defaults(func=cmd_txt2img)

    args = parser.parse_args(argv)
    from .errors import SdmuxError

    try:
        return args.func(args)
    except SdmuxError as e:
        print(f"error ({e.code}): {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
