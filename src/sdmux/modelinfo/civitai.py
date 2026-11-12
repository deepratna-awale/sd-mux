"""
Civitai's by-hash lookup and a hash cache. Nothing here runs unless the caller asked for it.

GET https://civitai.com/api/v1/model-versions/by-hash/{sha256} answers a model-version record:
id, modelId, name, baseModel ("SD 1.5", "Pony", "Illustrious", ...), baseModelType, trainedWords,
model {name, type, nsfw}, files[], air ("urn:air:sd1:checkpoint:civitai:<model>@<version>").
An unknown hash answers 404. (Checked with a real call on 2026-11-11.)
"""

from __future__ import annotations

import hashlib
import json
import os
import threading

CIVITAI_BY_HASH = "https://civitai.com/api/v1/model-versions/by-hash/{sha256}"
_lock = threading.Lock()


def cache_dir() -> str:
    """$SDMUX_CACHE_DIR, else $XDG_CACHE_HOME/sdmux, else ~/.cache/sdmux."""
    root = os.environ.get("SDMUX_CACHE_DIR")
    if root:
        return root
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "sdmux")


def _cache_file(name: str) -> str:
    return os.path.join(cache_dir(), name)


def _read_cache(name: str) -> dict:
    try:
        with open(_cache_file(name), "r", encoding="utf-8") as handle:
            data = json.load(handle)
            return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_cache(name: str, data: dict) -> None:
    path = _cache_file(name)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
        os.replace(tmp, path)
    except OSError:
        pass


def file_sha256(path: str, chunk: int = 8 * 1024 * 1024) -> str:
    """sha256 of a file, cached by absolute path, size and mtime (a 7 GB checkpoint takes several seconds to hash)."""
    path = os.path.abspath(path)
    stat = os.stat(path)
    key = f"{path}|{stat.st_size}|{int(stat.st_mtime)}"
    with _lock:
        cache = _read_cache("hashes.json")
        if key in cache:
            return cache[key]
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    value = digest.hexdigest()
    with _lock:
        cache = _read_cache("hashes.json")
        cache[key] = value
        _write_cache("hashes.json", cache)
    return value


def civitai_lookup(sha256: str | None, api_key: str | None = None, timeout: float = 15.0) -> dict | None:
    """The Civitai model-version record for a sha256, None when Civitai does not know it. Results are cached."""
    if not sha256:
        return None
    sha256 = sha256.lower()
    with _lock:
        cache = _read_cache("civitai.json")
        if sha256 in cache:
            return cache[sha256] or None
    import requests

    from .. import __version__

    headers = {"User-Agent": f"sdmux/{__version__}", "Accept": "application/json"}
    key = api_key or os.environ.get("CIVITAI_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        response = requests.get(CIVITAI_BY_HASH.format(sha256=sha256), headers=headers, timeout=timeout)
    except requests.RequestException:
        return None  # offline or blocked: not cached, so a later call can retry
    if response.status_code == 404:
        record = {}
    elif response.status_code == 200:
        try:
            record = response.json()
        except ValueError:
            return None
        record.pop("description", None)
        record.pop("images", None)
    else:
        return None
    with _lock:
        cache = _read_cache("civitai.json")
        cache[sha256] = record
        _write_cache("civitai.json", cache)
    return record or None
