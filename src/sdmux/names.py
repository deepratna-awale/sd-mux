"""
One vocabulary in, native vocabulary out.

Sampler and scheduler names are accepted in A1111 form ("Euler a", "DPM++ 2M",
"Karras", even the old combined "DPM++ 2M Karras") or in a backend's own form
("euler_ancestral", "dpmpp_2m", "karras"). Both are reduced to a canonical id
(ComfyUI's k-diffusion ids, which SwarmUI also uses) and then matched against
the names the server actually lists.

Model and LoRA names are matched loosely: folder, extension, case and an
"[hash]" suffix are ignored.
"""

from __future__ import annotations

import re

LORA_PATTERN = re.compile(r"<\s*lora\s*:([^:<>]+)(?::([^:<>]*))?(?::[^<>]*)?>", re.IGNORECASE)
MODEL_EXTENSIONS = r"safetensors|ckpt|pt|pth|bin|gguf|sft"


def stem(name: str) -> str:
    """'SDXL/foo [abc123].safetensors' -> 'foo' (lower case), for loose comparisons."""
    name = str(name).split("///", 1)[0]
    name = re.sub(r"\s*\[[0-9a-fA-F]{6,}\]\s*$", "", name)
    name = name.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    name = re.sub(rf"\.({MODEL_EXTENSIONS})$", "", name, flags=re.IGNORECASE)
    return name.strip().lower()


def lora_stem(name: str) -> str:
    """'SDXL/detail_tweaker.safetensors' -> 'detail_tweaker', the name <lora:...> tags use (case kept)."""
    name = str(name).replace("\\", "/").rsplit("/", 1)[-1]
    return re.sub(rf"\.({MODEL_EXTENSIONS})$", "", name, flags=re.IGNORECASE)


def match_choice(value: str | None, choices: list[str]) -> str | None:
    """
    Find value among the names a server offers, ignoring folders, extensions, case and "[hash]".

    An exact match wins, then a stem match, then a unique stem prefix ("waiIllustrious" finds
    "waiIllustriousSDXL_v170.safetensors" when nothing else starts that way).
    """
    if not value or not choices:
        return None
    if value in choices:
        return value
    wanted = stem(value)
    stems = [(choice, stem(choice)) for choice in choices]
    exact = [choice for choice, s in stems if s == wanted]
    if exact:
        return exact[0]
    # Full relative path given with a different separator or case.
    norm = value.replace("\\", "/").lower()
    for choice in choices:
        if choice.replace("\\", "/").lower() == norm:
            return choice
    prefixed = [choice for choice, s in stems if s.startswith(wanted)]
    if len(prefixed) == 1:
        return prefixed[0]
    return None


def find_choice(value: str | None, choices: list[str]) -> str | None:
    """match_choice(), then a unique choice containing value once spaces and punctuation are ignored ("anime6b")."""
    found = match_choice(value, choices)
    if found or not value:
        return found
    wanted = re.sub(r"[^a-z0-9]", "", value.lower())
    hits = [choice for choice in choices if wanted and wanted in re.sub(r"[^a-z0-9]", "", choice.lower())]
    return hits[0] if len(hits) == 1 else None


def extract_loras(prompt: str) -> tuple[str, list[tuple[str, float]]]:
    """
    Pull <lora:name:weight> tags out of a prompt.

    Returns the prompt without the tags and [(name, weight)]. A missing or unreadable weight is 1.0.
    """
    loras: list[tuple[str, float]] = []
    for name, weight in LORA_PATTERN.findall(prompt or ""):
        try:
            loras.append((name.strip(), float(weight) if weight.strip() else 1.0))
        except ValueError:
            loras.append((name.strip(), 1.0))
    cleaned = LORA_PATTERN.sub("", prompt or "")
    parts = [part.strip() for part in cleaned.split(",")]
    cleaned = ", ".join(part for part in parts if part)
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip(), loras


def lora_tags(loras: list[tuple[str, float]]) -> str:
    return " ".join(f"<lora:{name}:{weight:g}>" for name, weight in loras)


# Samplers -------------------------------------------------------------------
# canonical id -> A1111 label
SAMPLER_LABELS = {
    "euler": "Euler",
    "euler_ancestral": "Euler a",
    "heun": "Heun",
    "heunpp2": "Heun++ 2",
    "lms": "LMS",
    "dpm_2": "DPM2",
    "dpm_2_ancestral": "DPM2 a",
    "dpm_fast": "DPM fast",
    "dpm_adaptive": "DPM adaptive",
    "dpmpp_2s_ancestral": "DPM++ 2S a",
    "dpmpp_sde": "DPM++ SDE",
    "dpmpp_2m": "DPM++ 2M",
    "dpmpp_2m_sde": "DPM++ 2M SDE",
    "dpmpp_2m_sde_heun": "DPM++ 2M SDE Heun",
    "dpmpp_3m_sde": "DPM++ 3M SDE",
    "restart": "Restart",
    "ddim": "DDIM",
    "ddim_cfgpp": "DDIM CFG++",
    "plms": "PLMS",
    "uni_pc": "UniPC",
    "lcm": "LCM",
    "ddpm": "DDPM",
    "deis": "DEIS",
    "ipndm": "iPNDM",
    "ipndm_v": "iPNDM_V",
    "euler_cfg_pp": "Euler CFG++",
    "euler_ancestral_cfg_pp": "Euler a CFG++",
    "dpmpp_2m_cfg_pp": "DPM++ 2M CFG++",
    "tcd": "TCD",
}

_SAMPLER_EXTRA = {
    "euler a": "euler_ancestral",
    "euler_a": "euler_ancestral",
    "k_euler": "euler",
    "k_euler_a": "euler_ancestral",
    "k_euler_ancestral": "euler_ancestral",
    "unipc": "uni_pc",
    "dpm++ 2m sde gpu": "dpmpp_2m_sde",
    "dpm++ 3m sde gpu": "dpmpp_3m_sde",
    "kdpm_2": "dpm_2",
    "kdpm_2_a": "dpm_2_ancestral",
    "dpmpp_2s": "dpmpp_2s_ancestral",
    "dpmpp_3m": "dpmpp_3m_sde",
    "euler ancestral": "euler_ancestral",
    "dpm++ 2m sde exponential": "dpmpp_2m_sde",
}

# canonical id -> A1111 label
SCHEDULER_LABELS = {
    "automatic": "Automatic",
    "normal": "Normal",
    "karras": "Karras",
    "exponential": "Exponential",
    "polyexponential": "Polyexponential",
    "sgm_uniform": "SGM Uniform",
    "kl_optimal": "KL Optimal",
    "align_your_steps": "Align Your Steps",
    "simple": "Simple",
    "ddim_uniform": "DDIM",
    "beta": "Beta",
    "linear_quadratic": "Linear Quadratic",
    "uniform": "Uniform",
}
_SCHEDULER_EXTRA = {
    "ddim": "ddim_uniform",
    "sgm uniform": "sgm_uniform",
    "kl optimal": "kl_optimal",
    "align your steps": "align_your_steps",
    "ays": "align_your_steps",
    "default": "automatic",
    "auto": "automatic",
}

# Scheduler words that old A1111 versions glued onto sampler names ("DPM++ 2M Karras").
_COMBINED_SUFFIXES = ("karras", "exponential", "sgm uniform", "polyexponential", "simple", "beta")


def _key(name: str) -> str:
    name = str(name).split("///", 1)[0]
    return re.sub(r"\s+", " ", name.strip().lower())


def _alias_table(labels: dict[str, str], extra: dict[str, str]) -> dict[str, str]:
    table = {}
    for cid, label in labels.items():
        table[cid] = cid
        table[_key(label)] = cid
        table[cid.replace("_", " ")] = cid
    table.update({_key(k): v for k, v in extra.items()})
    return table


SAMPLER_ALIASES = _alias_table(SAMPLER_LABELS, _SAMPLER_EXTRA)
SCHEDULER_ALIASES = _alias_table(SCHEDULER_LABELS, _SCHEDULER_EXTRA)


def canonical_sampler(name: str | None) -> tuple[str | None, str | None]:
    """
    ("DPM++ 2M Karras") -> ("dpmpp_2m", "karras"); ("Euler a") -> ("euler_ancestral", None).

    Unknown names come back lower-cased as their own id, so a backend-specific sampler still matches itself.
    """
    if not name:
        return None, None
    key = _key(name)
    if key in SAMPLER_ALIASES:
        return SAMPLER_ALIASES[key], None
    for suffix in _COMBINED_SUFFIXES:
        if key.endswith(" " + suffix):
            base = key[: -len(suffix) - 1]
            if base in SAMPLER_ALIASES:
                return SAMPLER_ALIASES[base], SCHEDULER_ALIASES[suffix]
    return key, None


def canonical_scheduler(name: str | None) -> str | None:
    if not name:
        return None
    key = _key(name)
    return SCHEDULER_ALIASES.get(key, key)


def pick_native(value: str | None, choices: list[str], canonical) -> str | None:
    """
    The entry of choices (a server's own list) that means the same as value, or None.

    canonical maps any name to a comparable id (canonical_scheduler, or the sampler part of canonical_sampler).
    """
    if not value:
        return None
    if not choices:
        return None
    if value in choices:
        return value
    key = _key(value)
    for choice in choices:
        if _key(choice) == key:
            return choice
    wanted = canonical(value)
    for choice in choices:
        if canonical(choice) == wanted:
            return choice
    return None


def sampler_id(name: str | None) -> str | None:
    return canonical_sampler(name)[0]


def a1111_sampler_label(name: str | None) -> str | None:
    cid = sampler_id(name)
    return SAMPLER_LABELS.get(cid, name) if cid else None


def a1111_scheduler_label(name: str | None) -> str | None:
    cid = canonical_scheduler(name)
    return SCHEDULER_LABELS.get(cid, name) if cid else None


def split_option(value: str) -> tuple[str, str]:
    """SwarmUI lists some values as 'id///Display Name'. Returns (id, display)."""
    if "///" in value:
        ident, display = value.split("///", 1)
        return ident, display
    return value, value
