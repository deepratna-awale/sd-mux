"""
Model family (Pony, Illustrious, NoobAI, Animagine, ...) and other facts the tensors cannot tell.

Pony, Illustrious, NoobAI and Animagine all share the SDXL layout, so family comes from
outside the weights. Sources, most trusted first:

1. Sidecar files next to the model (free, local):
   <name>.metadata.json   ComfyUI LoRA Manager (sha256, base_model, civitai{})
   <name>.json            Civitai Browser+ / Civitai Helper (sha256, modelId, modelVersionId, "sd version")
   <name>.civitai.info    Civitai Helper (a full Civitai model-version record)
   <name>.civitai.json    Civitai model-version summary (modelId, versionId, baseModel)
   <name>.cm-info.json    StabilityMatrix (ModelId, VersionId, BaseModel, Hashes)
2. Civitai's by-hash lookup, only when asked (online=True).
3. The file's own __metadata__ (modelspec.*, ss_* training keys, merge recipes).
4. The file name, lowest confidence.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

# Civitai baseModel -> (architecture, family, variant)
CIVITAI_BASES = {
    "sd 1.4": ("sd15", None, None),
    "sd 1.5": ("sd15", None, None),
    "sd 1.5 lcm": ("sd15", None, "lcm"),
    "sd 1.5 hyper": ("sd15", None, "hyper"),
    "sd 2.0": ("sd2", None, None),
    "sd 2.0 768": ("sd2", None, None),
    "sd 2.1": ("sd2", None, None),
    "sd 2.1 768": ("sd2", None, None),
    "sd 2.1 unclip": ("sd2", None, None),
    "sdxl 0.9": ("sdxl", "sdxl", None),
    "sdxl 1.0": ("sdxl", "sdxl", None),
    "sdxl 1.0 lcm": ("sdxl", "sdxl", "lcm"),
    "sdxl turbo": ("sdxl", "sdxl", "turbo"),
    "sdxl lightning": ("sdxl", "sdxl", "lightning"),
    "sdxl hyper": ("sdxl", "sdxl", "hyper"),
    "sdxl distilled": ("sdxl", "sdxl", None),
    "pony": ("sdxl", "pony", None),
    "pony v7": ("auraflow", "pony", None),
    "illustrious": ("sdxl", "illustrious", None),
    "noobai": ("sdxl", "noobai", None),
    "playground v2": ("sdxl", "playground", None),
    "stable cascade": ("stable-cascade", None, None),
    "sd 3": ("sd3", None, None),
    "sd 3.5": ("sd3", None, None),
    "sd 3.5 large": ("sd3", None, None),
    "sd 3.5 large turbo": ("sd3", None, "turbo"),
    "sd 3.5 medium": ("sd3", None, None),
    "flux.1 d": ("flux", None, "dev"),
    "flux.1 s": ("flux", None, "schnell"),
    "flux.1 krea": ("flux", None, "dev"),
    "flux.1 kontext": ("flux", None, "kontext"),
    "flux.2 d": ("flux2", None, "dev"),
    "flux.2 klein": ("flux2", None, None),
    "chroma": ("chroma", None, None),
    "auraflow": ("auraflow", None, None),
    "hunyuan 1": ("hunyuan-dit", None, None),
    "hunyuan video": ("hunyuan-video", None, None),
    "hidream": ("hidream", None, None),
    "kolors": ("kolors", None, None),
    "lumina": ("lumina2", None, None),
    "pixart a": ("pixart", None, None),
    "pixart e": ("pixart", None, None),
    "qwen": ("qwen-image", None, None),
    "wan video": ("wan", None, None),
    "wan video 14b t2v": ("wan", None, None),
    "ltxv": ("ltxv", None, None),
    "mochi": ("mochi", None, None),
    "cogvideox": ("cogvideox", None, None),
    "z-image": ("lumina2", "z-image", None),
    "z-image turbo": ("lumina2", "z-image", "turbo"),
}

# "sd version" values written by Civitai Browser+ / Civitai Helper
SD_VERSION = {"sd1": "sd15", "sd2": "sd2", "sdxl": "sdxl", "sd3": "sd3", "flux": "flux"}

# Family from a name (file name, training base, merge recipe). Checked in order, first match wins.
# Pony comes before e621 because Pony merges often carry furry names too.
FAMILY_PATTERNS = [
    ("pony", r"pony|pdxl|autismmix|score_?9|ponyxl"),
    ("noobai", r"noob"),
    ("illustrious", r"illustrious|illu(?!sion)|\bwai|wai[-_ ]?(nsfw|illu|ani|real)|obsession|hassakuxl"),
    ("animagine", r"animagine"),
    ("furry", r"e621|furry|fluffy|fluffusion|yiff|bb95|indigo"),
    ("anime", r"anime|anything|counterfeit|meina|orangemix|abyss|\bnai\b|animefull|hassaku|anyhentai|aom|nai[-_]?full"),
]

# Architecture hints from a name, lowest confidence, used only when the file cannot be read.
ARCH_PATTERNS = [
    ("flux2", r"flux[.\-_ ]?2"),
    ("flux", r"flux"),
    ("sd3", r"sd[-_ ]?3|stable.?diffusion.?3"),
    ("chroma", r"chroma"),
    ("qwen-image", r"qwen"),
    ("hidream", r"hidream"),
    ("auraflow", r"auraflow"),
    ("lumina2", r"lumina|z[-_ ]?image"),
    ("wan", r"\bwan[-_ ]?2"),
    ("sdxl", r"xl|pony|illustrious|illu|noob|animagine|\bwai"),
    ("sd15", r"sd[-_ ]?1\.?5|v1-5|\b15\b"),
]


@dataclass
class Fact:
    """One fact from one source: field (architecture/family/variant/prediction/sha256/...), value, source, detail, confidence."""

    field: str
    value: object
    source: str
    detail: str = ""
    confidence: str = "medium"


def family_from_name(name: str | None) -> tuple[str, str] | None:
    """(family, pattern) from a free-text name, or None."""
    text = (name or "").lower()
    for family, pattern in FAMILY_PATTERNS:
        if re.search(pattern, text):
            return family, pattern
    return None


def architecture_from_name(name: str | None) -> tuple[str, str] | None:
    text = (name or "").lower()
    for arch, pattern in ARCH_PATTERNS:
        if re.search(pattern, text):
            return arch, pattern
    return None


def civitai_base(base: str | None) -> tuple[str | None, str | None, str | None] | None:
    if not base:
        return None
    key = str(base).strip().lower()
    if key in CIVITAI_BASES:
        return CIVITAI_BASES[key]
    if key in ("other", "unknown", ""):
        return None
    return None


def _facts_from_base(base, source, detail_prefix, confidence) -> list[Fact]:
    mapped = civitai_base(base)
    if not mapped:
        return []
    arch, family, variant = mapped
    detail = f"{detail_prefix}={base}"
    facts = [Fact("architecture", arch, source, detail, confidence)]
    if family:
        facts.append(Fact("family", family, source, detail, confidence))
    if variant:
        facts.append(Fact("variant", variant, source, detail, confidence))
    return facts


# Sidecars -------------------------------------------------------------------------
def sidecar_paths(path: str) -> list[tuple[str, str]]:
    """(format, path) of every sidecar that exists next to a model file."""
    base, _ = os.path.splitext(path)
    candidates = [
        ("lora-manager", base + ".metadata.json"),
        ("civitai-info", base + ".civitai.info"),
        ("civitai-json", base + ".civitai.json"),
        ("stability-matrix", base + ".cm-info.json"),
        ("civitai-helper", base + ".json"),
    ]
    return [(fmt, p) for fmt, p in candidates if os.path.isfile(p)]


def _load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError, UnicodeDecodeError):
        return None


def sidecar_facts(path: str) -> list[Fact]:
    facts: list[Fact] = []
    for fmt, sidecar in sidecar_paths(path):
        data = _load_json(sidecar)
        if not isinstance(data, dict):
            continue
        source = f"sidecar:{os.path.basename(sidecar)}"
        if fmt == "lora-manager":
            civ: dict = data["civitai"] if isinstance(data.get("civitai"), dict) else {}
            base = civ.get("baseModel") or (data.get("base_model") if data.get("base_model") not in (None, "Unknown") else None)
            facts += _facts_from_base(base, source, "base_model", "high")
            if data.get("sha256"):
                facts.append(Fact("sha256", str(data["sha256"]).lower(), source, "sha256", "high"))
            if civ.get("id") or civ.get("modelId"):
                facts.append(Fact("civitai", {"modelId": civ.get("modelId"), "versionId": civ.get("id")}, source, "", "high"))
        elif fmt in ("civitai-info", "civitai-json"):
            facts += _facts_from_base(data.get("baseModel"), source, "baseModel", "high")
            files = data.get("files") or []
            for f in files:
                sha = ((f or {}).get("hashes") or {}).get("SHA256")
                if sha:
                    facts.append(Fact("sha256", sha.lower(), source, "files[].hashes.SHA256", "medium"))
                    break
            model_id = data.get("modelId") or (data.get("model") or {}).get("id")
            version_id = data.get("versionId") or data.get("id")
            if model_id or version_id:
                facts.append(Fact("civitai", {"modelId": model_id, "versionId": version_id}, source, "", "high"))
        elif fmt == "stability-matrix":
            facts += _facts_from_base(data.get("BaseModel"), source, "BaseModel", "high")
            sha = (data.get("Hashes") or {}).get("SHA256")
            if sha:
                facts.append(Fact("sha256", sha.lower(), source, "Hashes.SHA256", "high"))
            if data.get("ModelId") or data.get("VersionId"):
                facts.append(Fact("civitai", {"modelId": data.get("ModelId"), "versionId": data.get("VersionId")}, source, "", "high"))
        elif fmt == "civitai-helper":
            if not any(k in data for k in ("sha256", "sd version", "modelId", "activation text")):
                continue  # some other tool's .json; not a model sidecar
            version = SD_VERSION.get(str(data.get("sd version", "")).strip().lower())
            if version:
                facts.append(Fact("architecture", version, source, f"sd version={data.get('sd version')}", "medium"))
            if data.get("sha256"):
                facts.append(Fact("sha256", str(data["sha256"]).lower(), source, "sha256", "high"))
            if data.get("modelId") or data.get("modelVersionId"):
                facts.append(Fact("civitai", {"modelId": data.get("modelId"), "versionId": data.get("modelVersionId")}, source, "", "high"))
    return facts


# A1111 hash cache ----------------------------------------------------------------
def a1111_cached_sha256(path: str) -> str | None:
    """
    sha256 from an A1111/Forge install's cache/hashes (a diskcache SQLite store), when the model sits in one.

    The cached value is a pickled dict; it is read with the same data-only pickle walker as .ckpt files.
    """
    import sqlite3

    from .headers import _walk_pickle

    folder = os.path.dirname(os.path.abspath(path))
    name = os.path.basename(path)
    stem = os.path.splitext(name)[0]
    root = folder
    for _ in range(4):
        db = os.path.join(root, "cache", "hashes", "cache.db")
        if os.path.isfile(db):
            break
        root = os.path.dirname(root)
    else:
        return None
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=1)
        try:
            rows = con.execute("select key, value from Cache where key like ? or key like ?",
                               (f"%/{name}", f"%/{stem}")).fetchall()
        finally:
            con.close()
    except sqlite3.Error:
        return None
    mtime = os.path.getmtime(path)
    for _key, value in rows:
        if not isinstance(value, (bytes, bytearray)):
            continue
        try:
            data = _walk_pickle(bytes(value))
        except Exception:
            continue
        if isinstance(data, dict) and data.get("sha256") and abs(float(data.get("mtime") or 0) - mtime) < 1.0:
            return str(data["sha256"]).lower()
    return None


# __metadata__ --------------------------------------------------------------------
MODELSPEC_ARCH = [
    ("stable-diffusion-xl-v1-refiner", "sdxl-refiner"),
    ("stable-diffusion-xl", "sdxl"),
    ("stable-diffusion-v1", "sd15"),
    ("stable-diffusion-v2", "sd2"),
    ("stable-diffusion-3", "sd3"),
    ("flux-1", "flux"),
    ("flux.1", "flux"),
    ("flux-2", "flux2"),
    ("chroma", "chroma"),
    ("qwen-image", "qwen-image"),
    ("lumina", "lumina2"),
    ("hidream", "hidream"),
    ("wan", "wan"),
]
SS_BASE = {
    "sd_v1": "sd15",
    "sd_v2": "sd2",
    "sd_v2_768_v": "sd2",
    "sdxl_base_v0-9": "sdxl",
    "sdxl_base_v1-0": "sdxl",
    "sdxl_refiner_v1-0": "sdxl-refiner",
    "sd3": "sd3",
    "flux1": "flux",
}


def metadata_facts(metadata: dict) -> list[Fact]:
    facts: list[Fact] = []
    if not metadata:
        return facts
    spec_arch = str(metadata.get("modelspec.architecture") or "").lower()
    if spec_arch:
        for prefix, arch in MODELSPEC_ARCH:
            if spec_arch.startswith(prefix):
                facts.append(Fact("architecture", arch, "metadata", f"modelspec.architecture={spec_arch}", "medium"))
                break
        if "inpaint" in spec_arch:
            facts.append(Fact("variant", "inpaint", "metadata", f"modelspec.architecture={spec_arch}", "medium"))
        if "dev" in spec_arch and "flux" in spec_arch:
            facts.append(Fact("variant", "dev", "metadata", f"modelspec.architecture={spec_arch}", "medium"))
        if "schnell" in spec_arch:
            facts.append(Fact("variant", "schnell", "metadata", f"modelspec.architecture={spec_arch}", "medium"))
    base_version = str(metadata.get("ss_base_model_version") or "").lower()
    if base_version in SS_BASE:
        facts.append(Fact("architecture", SS_BASE[base_version], "metadata", f"ss_base_model_version={base_version}", "medium"))
    elif base_version.startswith("flux"):
        facts.append(Fact("architecture", "flux", "metadata", f"ss_base_model_version={base_version}", "medium"))
    for key in ("modelspec.title", "ss_sd_model_name", "ss_base_model", "modelspec.description"):
        hit = family_from_name(metadata.get(key))
        if hit:
            facts.append(Fact("family", hit[0], "metadata", f"{key}={str(metadata.get(key))[:80]} matches /{hit[1]}/",
                              "medium" if key != "modelspec.description" else "low"))
            break
    for key in ("sd_merge_recipe", "sd_merge_models"):
        raw = metadata.get(key)
        if not raw:
            continue
        names = _merge_names(raw)
        families = [f for f in (family_from_name(n) for n in names) if f]
        if families:
            counts: dict[str, int] = {}
            for fam, _ in families:
                counts[fam] = counts.get(fam, 0) + 1
            fam = max(counts, key=lambda f: counts[f])
            facts.append(Fact("family", fam, "metadata", f"{key} lists {', '.join(names[:4])}", "low"))
            break
    if metadata.get("ss_sd_model_name"):
        facts.append(Fact("trained_on", metadata["ss_sd_model_name"], "metadata", "ss_sd_model_name", "high"))
    sha = metadata.get("modelspec.hash_sha256")
    if sha and re.fullmatch(r"(0x)?[0-9a-fA-F]{64}", str(sha)):
        facts.append(Fact("weights_sha256", str(sha).lower().removeprefix("0x"), "metadata", "modelspec.hash_sha256", "medium"))
    return facts


def _merge_names(raw) -> list[str]:
    """Model names out of an A1111 merge recipe / supermerger model list (JSON strings of varying shape)."""
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return [str(raw)]
    names: list[str] = []

    def walk(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in ("name", "legacy_hash", "sha256") and isinstance(value, str):
                    if key == "name":
                        names.append(value)
                elif key.startswith("primary_model") or key.startswith("secondary_model") or key.startswith("tertiary_model"):
                    if isinstance(value, str):
                        names.append(value)
                else:
                    walk(value)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(data)
    return names
