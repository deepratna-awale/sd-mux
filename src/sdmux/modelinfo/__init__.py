"""
Checkpoint identification: what a model file is, from its header and what sits next to it.

    info = sdmux.identify("model.safetensors")       # reads only the header, never the weights
    info.kind          checkpoint | diffusion_model | lora | vae | controlnet | text_encoder | embedding | unknown
    info.architecture  sd15 | sd2 | sdxl | sdxl-refiner | sd3 | flux | flux2 | chroma | qwen-image | wan | ...
    info.family        pony | illustrious | noobai | animagine | anime | furry | sdxl | ... (None when unknown)
    info.prediction    epsilon | v | flow | edm | None
    info.variant       inpaint | instruct | dev | schnell | fill | turbo | ... | None
    info.evidence      which rule or source decided each field, and how confident
    info.profile       prompt dialect + starting defaults

identify(path, online=True) adds Civitai's by-hash lookup. Nothing touches the network otherwise.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from . import family as fam
from .civitai import civitai_lookup, file_sha256
from .headers import Header, HeaderError, read_header
from .lookups import Link, env_sources, lookup_all
from .profiles import Profile, build_profile
from .rules import analyse, ztsnr
from .rules import prediction as find_prediction

CONFIDENCE = {"high": 3, "medium": 2, "low": 1}
FIELDS = ("kind", "architecture", "family", "prediction", "variant")


@dataclass
class Evidence:
    field: str
    value: object
    source: str  # header | sidecar:<file> | civitai | civarchive:<platform> | huggingface | modelindex | metadata | filename | backend
    detail: str
    confidence: str  # high | medium | low

    def __str__(self) -> str:
        return f"{self.field}={self.value} ({self.confidence}, {self.source}: {self.detail})"


@dataclass
class ModelInfo:
    path: str | None
    name: str
    kind: str = "unknown"
    architecture: str | None = None
    family: str | None = None
    prediction: str | None = None
    variant: str | None = None
    format: str | None = None
    size: int | None = None
    sha256: str | None = None
    trained_on: str | None = None
    zero_terminal_snr: bool = False
    text_encoder: str | None = None
    civitai: dict | None = None
    links: list = field(default_factory=list)  # Link: where the file is published (online lookups)
    metadata: dict = field(default_factory=dict)
    evidence: list[Evidence] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)

    @property
    def profile(self) -> Profile:
        return build_profile(self.architecture, self.family, self.variant, self.prediction)

    def why(self, field_name: str) -> list[Evidence]:
        return [e for e in self.evidence if e.field == field_name]

    def to_dict(self, metadata: bool = False) -> dict:
        out = {
            "path": self.path, "name": self.name, "kind": self.kind, "architecture": self.architecture,
            "family": self.family, "prediction": self.prediction, "variant": self.variant, "format": self.format,
            "size": self.size, "sha256": self.sha256, "trained_on": self.trained_on,
            "zero_terminal_snr": self.zero_terminal_snr, "text_encoder": self.text_encoder, "civitai": self.civitai,
            "links": [link.__dict__ for link in self.links],
            "evidence": [e.__dict__ for e in self.evidence], "conflicts": self.conflicts,
            "profile": self.profile.to_dict() if self.kind in ("checkpoint", "diffusion_model", "unknown") else None,
        }
        if metadata:
            out["metadata"] = self.metadata
        return out


def _decide(info: ModelInfo, candidates: list[Evidence]) -> None:
    """Pick each field's value from the most trusted evidence; record disagreements."""
    order = {"header": 5, "civitai": 4, "sidecar": 4, "civarchive": 4, "backend": 3, "huggingface": 3, "metadata": 2,
             "modelindex": 2, "filename": 1}

    def rank(e: Evidence):
        source = e.source.split(":", 1)[0]
        return (order.get(source, 0), CONFIDENCE.get(e.confidence, 0))

    for name in ("architecture", "family", "variant", "prediction"):
        options = [e for e in candidates if e.field == name and e.value is not None]
        if not options:
            continue
        best = max(options, key=rank)
        # A sidecar or Civitai "family" outranks the header (which has no family) and the name.
        setattr(info, name, best.value)
        info.evidence.append(best)
        for other in options:
            if other.value != best.value and other is not best:
                info.conflicts.append(f"{name}: {other.value} from {other.source} ({other.detail}) disagrees; kept {best.value}")


def identify(path, online: bool = False, *, hash_file: bool | None = None, civitai_key: str | None = None,
             sources: tuple[str, ...] | None = None, timeout: float = 15.0) -> ModelInfo:
    """
    Identify a model file. Reads the header (safetensors, GGUF, or a torch .ckpt walked without unpickling),
    sidecar files next to it, its embedded metadata and its name.

    online=True also looks the sha256 up on Civitai (civitai.com, then civitai.red), CivArchive (Tensor.Art, SeaArt,
    Shakker, PixAI, deleted Civitai models, ...), modelindex.dev (Hugging Face / ModelScope) and the Hugging Face model
    card. sources limits which (default: all, or $SDMUX_LOOKUPS). The hash comes from a sidecar or an A1111 hash cache when one
    exists; otherwise the file is hashed (cached by path, size and mtime). hash_file=True hashes even offline.
    """
    path = os.fspath(path)
    name = os.path.basename(path)
    info = ModelInfo(path=path, name=name, size=os.path.getsize(path))
    candidates: list[Evidence] = []

    header: Header | None = None
    try:
        header = read_header(path)
    except (HeaderError, OSError, ValueError) as e:
        info.conflicts.append(f"header unreadable: {e}")
    if header is not None:
        info.format = header.format
        info.metadata = dict(header.metadata)
        structure = analyse(header.tensors, header.metadata)
        info.kind = structure.kind
        info.evidence.append(Evidence("kind", structure.kind, "header", _kind_detail(structure, header), "high"))
        info.text_encoder = structure.subtype
        if structure.architecture and structure.architecture.value:
            a = structure.architecture
            candidates.append(Evidence("architecture", a.value, "header", a.rule, a.confidence))
        if structure.variant:
            v = structure.variant
            candidates.append(Evidence("variant", v.value, "header", v.rule, v.confidence))
        if header.format == "gguf":
            arch = str(header.metadata.get("general.architecture") or "")
            mapped = {"sd1": "sd15", "sdxl": "sdxl", "flux": "flux", "sd3": "sd3", "wan": "wan", "ltxv": "ltxv",
                      "hyvid": "hunyuan-video", "lumina2": "lumina2", "qwen_image": "qwen-image", "hidream": "hidream",
                      "chroma": "chroma", "cosmos": "cosmos"}.get(arch)
            if mapped:
                candidates.append(Evidence("architecture", mapped, "metadata", f"general.architecture={arch}", "medium"))
        info.zero_terminal_snr = ztsnr(header.tensors)
        for fact in fam.metadata_facts(header.metadata):
            _add_fact(info, candidates, fact)
    else:
        candidates.extend(_name_evidence(name, "architecture"))

    for fact in fam.sidecar_facts(path):
        _add_fact(info, candidates, fact)

    if not info.sha256:
        cached = fam.a1111_cached_sha256(path)
        if cached:
            info.sha256 = cached
            info.evidence.append(Evidence("sha256", cached, "a1111-cache", "cache/hashes", "high"))

    if online or hash_file:
        if not info.sha256:
            info.sha256 = file_sha256(path)
            info.evidence.append(Evidence("sha256", info.sha256, "computed", "sha256 of the file", "high"))
        if online:
            _apply_lookups(info, candidates, info.sha256, sources, civitai_key, timeout)

    candidates.extend(_name_evidence(name, "family"))
    if info.trained_on and info.kind == "lora":
        hit = fam.family_from_name(info.trained_on)
        if hit:
            candidates.append(Evidence("family", hit[0], "metadata", f"trained on {info.trained_on} (matches /{hit[1]}/)", "medium"))

    _decide(info, candidates)
    _finish(info, header)
    return info


def identify_name(name: str, *, architecture: str | None = None, kind: str = "checkpoint", sha256: str | None = None,
                  online: bool = False, timeout: float = 15.0) -> ModelInfo:
    """Identify from a name alone (a model on a remote backend). architecture is what the backend reports, if anything."""
    info = ModelInfo(path=None, name=name, kind=kind or "unknown", sha256=sha256)
    candidates: list[Evidence] = []
    if architecture:
        candidates.append(Evidence("architecture", architecture, "backend", "reported by the backend", "high"))
    candidates.extend(_name_evidence(name, "architecture"))
    candidates.extend(_name_evidence(name, "family"))
    if online and sha256:
        _apply_lookups(info, candidates, sha256, None, None, timeout)
    _decide(info, candidates)
    _finish(info, None)
    return info


def _apply_lookups(info, candidates, sha256, sources, civitai_key, timeout) -> None:
    for result in lookup_all(sha256, tuple(sources) if sources else env_sources(), timeout, civitai_key):
        for fact in result.facts:
            candidates.append(Evidence(fact.field, fact.value, fact.source, fact.detail, fact.confidence))
        for link in result.links:
            if all(link.url != known.url for known in info.links):
                info.links.append(link)
        if result.service == "civitai" and result.record:
            r = result.record
            info.civitai = {"modelId": r.get("modelId"), "versionId": r.get("id"), "name": r.get("name"),
                            "model": (r.get("model") or {}).get("name"), "baseModel": r.get("baseModel"),
                            "trainedWords": r.get("trainedWords"), "site": r.get("_host")}


def _add_fact(info: ModelInfo, candidates: list[Evidence], fact) -> None:
    if fact.field in ("architecture", "family", "variant", "prediction"):
        candidates.append(Evidence(fact.field, fact.value, fact.source, fact.detail, fact.confidence))
    elif fact.field == "sha256" and not info.sha256:
        info.sha256 = fact.value
        info.evidence.append(Evidence("sha256", fact.value, fact.source, fact.detail, fact.confidence))
    elif fact.field == "civitai" and not info.civitai:
        info.civitai = fact.value
    elif fact.field == "trained_on":
        info.trained_on = fact.value


def _name_evidence(name: str, field_name: str) -> list[Evidence]:
    if field_name == "family":
        hit = fam.family_from_name(name)
        return [Evidence("family", hit[0], "filename", f"name matches /{hit[1]}/", "low")] if hit else []
    hit = fam.architecture_from_name(name)
    return [Evidence("architecture", hit[0], "filename", f"name matches /{hit[1]}/", "low")] if hit else []


def _kind_detail(structure, header: Header) -> str:
    if structure.kind == "text_encoder":
        return f"text encoder layout ({structure.subtype})"
    return f"{len(header.tensors)} tensors in {header.format} header"


def _finish(info: ModelInfo, header: Header | None) -> None:
    """Fill prediction, and keep family consistent with architecture."""
    # A family only makes sense on its architecture: an SDXL family name on an SD1.5 file is a coincidence.
    sdxl_families = {"pony", "illustrious", "noobai", "animagine", "sdxl", "playground"}
    if info.family in sdxl_families and info.architecture not in (None, "sdxl", "sdxl-refiner", "auraflow"):
        fam_ev = next((e for e in info.evidence if e.field == "family"), None)
        if fam_ev and fam_ev.source in ("filename", "metadata"):
            info.conflicts.append(f"family {info.family} ({fam_ev.source}) does not fit {info.architecture}; dropped")
            info.evidence.remove(fam_ev)
            info.family = None
    if info.family is None and info.architecture == "sdxl" and info.kind in ("checkpoint", "diffusion_model"):
        pass  # unknown SDXL fine-tune: leave None rather than guess "sdxl"
    if header is not None and info.prediction is None and info.kind in ("checkpoint", "diffusion_model"):
        found = find_prediction(header.tensors, info.architecture, header.metadata)
        if found:
            info.prediction = found.value
            source = "header" if found.confidence == "high" and "modelspec" not in found.rule else "metadata"
            if "default" in found.rule or "flow-matching" in found.rule:
                source = "header"
            info.evidence.append(Evidence("prediction", found.value, source, found.rule, found.confidence))
    elif header is None and info.prediction is None and info.architecture in ("sd15", "sdxl"):
        info.prediction = "epsilon"
        info.evidence.append(Evidence("prediction", "epsilon", "filename", f"{info.architecture} default", "low"))
    if info.kind == "lora" and info.family is None and info.trained_on:
        pass


__all__ = ["identify", "identify_name", "ModelInfo", "Evidence", "Profile", "build_profile", "read_header", "HeaderError",
           "Link", "civitai_lookup"]
