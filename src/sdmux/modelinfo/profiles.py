"""Prompt dialect and starting defaults per architecture, family, variant and prediction type."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from importlib import resources
from typing import Any

DIALECTS = ("pony", "danbooru", "e621", "clip", "natural")


@lru_cache(maxsize=1)
def _table() -> dict:
    return json.loads(resources.files("sdmux").joinpath("resources/profiles.json").read_text(encoding="utf-8"))


@dataclass
class Profile:
    """
    How to prompt a model and where to start. These are starting points, not tuned truth.

    dialect: pony (score_ tags), danbooru (booru tags), e621 (e621 tags), clip (short phrases), natural (sentences).
    """

    dialect: str = "clip"
    width: int = 512
    height: int = 512
    steps: int = 25
    cfg: float = 7.0
    sampler: str | None = None
    scheduler: str | None = None
    clip_skip: int = 1
    quality: str = ""
    negative: str = ""
    guidance: float | None = None
    notes: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    @property
    def defaults(self) -> dict:
        """Keyword arguments for txt2img(): width, height, steps, cfg, sampler, scheduler, clip_skip."""
        out: dict[str, Any] = {"width": self.width, "height": self.height, "steps": self.steps, "cfg": self.cfg, "clip_skip": self.clip_skip}
        if self.sampler:
            out["sampler"] = self.sampler
        if self.scheduler and self.scheduler.lower() != "automatic":
            out["scheduler"] = self.scheduler
        return out

    def prompt(self, text: str) -> str:
        """text with the quality tags in front, skipping tags already present."""
        return _join(_missing(text, self.quality), text)

    def negative_prompt(self, text: str = "") -> str:
        return _join(text, _missing(text, self.negative))

    def to_dict(self) -> dict:
        return asdict(self)


def _missing(text: str, tags: str) -> str:
    present = {t.strip().lower() for t in (text or "").split(",")}
    return ", ".join(t.strip() for t in (tags or "").split(",") if t.strip() and t.strip().lower() not in present)


def _join(*parts: str) -> str:
    return ", ".join(p.strip().strip(",").strip() for p in parts if p and p.strip().strip(","))


def build_profile(architecture: str | None, family: str | None, variant: str | None = None, prediction: str | None = None) -> Profile:
    table = _table()
    values = dict(table["default"])
    sources = ["default"]
    notes: list[str] = []

    def apply(entry: dict | None, label: str):
        if not entry:
            return
        entry = dict(entry)
        note = entry.pop("note", None)
        if note:
            notes.append(note)
        values.update(entry)
        sources.append(label)

    apply(table["architectures"].get(architecture or ""), f"architecture:{architecture}")
    apply(table["families"].get(family or ""), f"family:{family}")
    apply(table["variants"].get(variant or ""), f"variant:{variant}")
    if prediction == "v":
        apply(table["predictions"]["v"], "prediction:v")
    values.pop("note", None)
    return Profile(
        dialect=values.get("dialect", "clip"),
        width=int(values["width"]),
        height=int(values["height"]),
        steps=int(values["steps"]),
        cfg=float(values["cfg"]),
        sampler=values.get("sampler"),
        scheduler=values.get("scheduler"),
        clip_skip=int(values.get("clip_skip") or 1),
        quality=values.get("quality") or "",
        negative=values.get("negative") or "",
        guidance=values.get("guidance"),
        notes=notes,
        sources=sources,
    )
