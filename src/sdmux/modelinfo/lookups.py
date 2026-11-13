"""
Online lookups by sha256 across the places people upload models. Nothing here runs unless asked (online=True).

Each lookup turns one service's answer into Facts (architecture, family, variant) plus Links (where the file is
hosted). Answers are cached per service and hash; "not found" is cached for $SDMUX_NEGATIVE_TTL seconds (default
7 days), network failures are not cached.

    civitai      GET {civitai.com, civitai.red}/api/v1/model-versions/by-hash/{sha256}
                 Model-version record with baseModel ("Pony", "Illustrious", "SD 1.5", "Flux.1 D", ...).
                 civitai.red (Civitai's mature-content site since April 2026) serves the same API and accounts;
                 it is asked when civitai.com does not know the hash.
    civarchive   GET civarchive.com/api/sha256/{sha256}
                 An archive that tracks models across Civitai, Tensor.Art, SeaArt, Shakker, PixAI, Hugging Face and
                 others, including models deleted from Civitai. Gives baseModel, the platform, and mirrors.
    modelindex   GET modelindex.dev/api/v1/hash/sha256:{sha256}
                 Which Hugging Face and ModelScope repositories host the file.
    huggingface  GET huggingface.co/api/models/{repo}
                 For Hugging Face matches from modelindex: the model card's base_model lineage and diffusers pipeline.

Response shapes were checked with real calls on 2026-11-12.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field

from . import family as fam
from .civitai import _cached_answer, _miss, _read_cache, _write_cache, civitai_lookup

_lock = threading.Lock()
SOURCES = ("civitai", "civarchive", "modelindex", "huggingface")
DEFAULT_SOURCES = SOURCES

CIVITAI_HOSTS = ("https://civitai.com", "https://civitai.red")
CIVARCHIVE = "https://civarchive.com/api/sha256/{sha256}"
MODELINDEX = "https://modelindex.dev/api/v1/hash/sha256:{sha256}"
HF_MODEL = "https://huggingface.co/api/models/{repo}"

# diffusers pipeline class (Hugging Face "diffusers:<Class>" tag) -> architecture
DIFFUSERS_PIPELINES = {
    "StableDiffusionPipeline": "sd15",
    "StableDiffusionXLPipeline": "sdxl",
    "StableDiffusionXLImg2ImgPipeline": "sdxl-refiner",
    "StableDiffusion3Pipeline": "sd3",
    "FluxPipeline": "flux",
    "Flux2Pipeline": "flux2",
    "ChromaPipeline": "chroma",
    "QwenImagePipeline": "qwen-image",
    "Lumina2Pipeline": "lumina2",
    "ZImagePipeline": "lumina2",
    "HiDreamImagePipeline": "hidream",
    "AuraFlowPipeline": "auraflow",
    "PixArtSigmaPipeline": "pixart",
    "PixArtAlphaPipeline": "pixart",
    "WanPipeline": "wan",
    "StableCascadeCombinedPipeline": "stable-cascade",
    "HunyuanDiTPipeline": "hunyuan-dit",
}


@dataclass
class Link:
    """Where a file with this hash is published."""

    service: str  # civitai | tensorart | seaart | shakker | pixai | huggingface | modelscope | ...
    url: str
    name: str | None = None
    model_id: str | None = None
    version_id: str | None = None


@dataclass
class Lookup:
    service: str
    found: bool
    facts: list = field(default_factory=list)  # family.Fact
    links: list[Link] = field(default_factory=list)
    record: dict | None = None


def _get_json(url: str, timeout: float, headers: dict | None = None):
    """(status, json or None). Network failures raise requests.RequestException."""
    import requests

    from .. import __version__

    h = {"User-Agent": f"sdmux/{__version__}", "Accept": "application/json", **(headers or {})}
    response = requests.get(url, headers=h, timeout=timeout)
    try:
        return response.status_code, response.json()
    except ValueError:
        return response.status_code, None


def _cached(service: str, key: str, fetch, missing=lambda value: not value):
    """Run fetch() once per (service, key); cache found answers, not-found ones for a while, never errors."""
    cache_key = f"{service}:{key}"
    with _lock:
        usable, value = _cached_answer(_read_cache("lookups.json").get(cache_key))
        if usable:
            return value
    try:
        value = fetch()
    except Exception:  # offline, blocked, malformed: try again next time
        return None
    with _lock:
        cache = _read_cache("lookups.json")
        cache[cache_key] = _miss(value) if missing(value) else value
        _write_cache("lookups.json", cache)
    return value


# Services --------------------------------------------------------------------------
def lookup_civitai(sha256: str, timeout: float = 15.0, api_key: str | None = None) -> Lookup:
    record = civitai_lookup(sha256, api_key=api_key, timeout=timeout, hosts=CIVITAI_HOSTS)
    if not record:
        return Lookup("civitai", False)
    host = record.get("_host") or CIVITAI_HOSTS[0]
    facts = fam._facts_from_base(record.get("baseModel"), "civitai", "baseModel", "high")
    model = record.get("model") or {}
    if not any(f.field == "family" for f in facts):
        hit = fam.family_from_name(model.get("name"))
        if hit:
            facts.append(fam.Fact("family", hit[0], "civitai", f"model name {model.get('name')!r} matches /{hit[1]}/", "medium"))
    link = Link("civitai", f"{host}/models/{record.get('modelId')}?modelVersionId={record.get('id')}", model.get("name"),
                str(record.get("modelId")), str(record.get("id")))
    return Lookup("civitai", True, facts, [link], record)


def lookup_civarchive(sha256: str, timeout: float = 15.0) -> Lookup:
    def fetch():
        status, body = _get_json(CIVARCHIVE.format(sha256=sha256), timeout)
        if status == 404:
            return {}
        if status != 200 or not isinstance(body, dict):
            raise RuntimeError(f"civarchive answered {status}")
        return body

    body = _cached("civarchive", sha256, fetch, missing=lambda b: not (b or {}).get("model"))
    if not body or not body.get("model"):
        # Without a "model" the archive only lists stray files that happen to share the hash.
        return Lookup("civarchive", False, record=body)
    model = body["model"]
    version = model.get("version") or {}
    platform = model.get("platform") or "unknown"
    source = f"civarchive:{platform}"
    facts = fam._facts_from_base(version.get("baseModel"), source, "baseModel", "high")
    if not any(f.field == "family" for f in facts):
        hit = fam.family_from_name(model.get("name"))
        if hit:
            facts.append(fam.Fact("family", hit[0], source, f"model name {model.get('name')!r} matches /{hit[1]}/", "medium"))
    links = []
    if version.get("platform_url"):
        links.append(Link(platform, version["platform_url"], model.get("name"), str(model.get("id")), str(version.get("id"))))
    for mirror in version.get("mirrors") or []:
        if mirror.get("platform_url"):
            links.append(Link(mirror.get("platform") or "unknown", mirror["platform_url"], mirror.get("name")))
    for f in body.get("files") or []:
        if f.get("source") and f.get("url") and f["source"] not in ("civitai",):
            links.append(Link(f["source"], f["url"], f.get("filename")))
    return Lookup("civarchive", True, facts, _dedupe(links), body)


def lookup_modelindex(sha256: str, timeout: float = 15.0) -> Lookup:
    def fetch():
        status, body = _get_json(MODELINDEX.format(sha256=sha256), timeout)
        if status == 404:
            return {}
        if status != 200 or not isinstance(body, dict):
            raise RuntimeError(f"modelindex answered {status}")
        return body

    body = _cached("modelindex", sha256, fetch)
    matches = (body or {}).get("matches") or []
    if not matches:
        return Lookup("modelindex", False)
    links, facts = [], []
    for m in matches:
        service = {"hf": "huggingface"}.get(m.get("source"), m.get("source") or "unknown")
        repo = f"{m.get('org')}/{m.get('name')}"
        base = "https://huggingface.co" if service == "huggingface" else "https://modelscope.cn/models"
        links.append(Link(service, f"{base}/{repo}", repo))
        hit = fam.family_from_name(repo + " " + str(m.get("path") or ""))
        if hit and not facts:
            facts.append(fam.Fact("family", hit[0], "modelindex", f"hosted as {repo}/{m.get('path')} (matches /{hit[1]}/)", "medium"))
    return Lookup("modelindex", True, facts, links, body)


def lookup_huggingface(repos: list[str], timeout: float = 15.0) -> Lookup:
    """Model-card facts for Hugging Face repos already known to host the file (from modelindex)."""
    facts = []
    for repo in repos[:3]:
        def fetch(repo=repo):
            status, body = _get_json(HF_MODEL.format(repo=repo), timeout)
            if status in (401, 404):
                return {}
            if status != 200 or not isinstance(body, dict):
                raise RuntimeError(f"huggingface answered {status}")
            card = body.get("cardData") or {}
            return {"base_model": card.get("base_model"), "tags": [t for t in body.get("tags") or []
                                                                    if t.startswith(("diffusers:", "base_model:"))]}

        data = _cached("huggingface", repo, fetch) or {}
        for tag in data.get("tags") or []:
            if tag.startswith("diffusers:"):
                arch = DIFFUSERS_PIPELINES.get(tag.split(":", 1)[1])
                if arch:
                    facts.append(fam.Fact("architecture", arch, "huggingface", f"{repo}: {tag}", "medium"))
        lineage = data.get("base_model") or []
        lineage = [lineage] if isinstance(lineage, str) else lineage
        for name in [repo, *lineage]:
            hit = fam.family_from_name(name)
            if hit:
                facts.append(fam.Fact("family", hit[0], "huggingface", f"{repo} (base_model {', '.join(lineage) or '-'}) matches /{hit[1]}/",
                                      "medium"))
                break
        if facts:
            break
    return Lookup("huggingface", bool(facts), facts)


def lookup_all(sha256: str, sources=DEFAULT_SOURCES, timeout: float = 15.0, civitai_key: str | None = None) -> list[Lookup]:
    """Ask each requested service in turn. Hugging Face is asked only about repos modelindex found."""
    sha256 = sha256.lower()
    results: list[Lookup] = []
    if "civitai" in sources:
        results.append(lookup_civitai(sha256, timeout, civitai_key))
    if "civarchive" in sources:
        results.append(lookup_civarchive(sha256, timeout))
    if "modelindex" in sources:
        mi = lookup_modelindex(sha256, timeout)
        results.append(mi)
        repos = [l.name for l in mi.links if l.service == "huggingface" and l.name]
        if "huggingface" in sources and repos:
            results.append(lookup_huggingface(repos, timeout))
    return results


def _dedupe(links: list[Link]) -> list[Link]:
    seen, out = set(), []
    for link in links:
        if link.url.startswith("/"):
            link.url = "https://civarchive.com" + link.url  # the archive's own page for that platform
        if link.url not in seen:
            seen.add(link.url)
            out.append(link)
    return out


def env_sources() -> tuple[str, ...]:
    """$SDMUX_LOOKUPS (comma list) overrides which services online lookups ask."""
    raw = os.environ.get("SDMUX_LOOKUPS")
    if not raw:
        return DEFAULT_SOURCES
    return tuple(s.strip() for s in raw.split(",") if s.strip() in SOURCES)
