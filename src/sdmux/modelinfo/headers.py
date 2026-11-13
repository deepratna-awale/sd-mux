"""
Read tensor names, shapes and metadata from a model file without loading its weights.

    .safetensors  8-byte little-endian header length, then that many bytes of JSON.
    .gguf         a binary header of key/value metadata and tensor infos (dims stored in reverse order).
    .ckpt / .pt   a zip holding a pickle. The pickle is never unpickled: its opcodes are walked by a
                  tiny stack machine that only builds strings, numbers, tuples, lists and dicts, so no
                  code in the file can run. Legacy non-zip pickles are read the same way.
"""

from __future__ import annotations

import json
import os
import pickletools
import struct
import zipfile
from dataclasses import dataclass, field

MAX_HEADER = 100 * 1024 * 1024  # safetensors headers on real files run to a few hundred KB


class HeaderError(ValueError):
    """The file is not a model format sdmux can read, or its header is damaged."""


@dataclass
class Header:
    format: str  # safetensors | gguf | ckpt
    tensors: dict[str, tuple[int, ...] | None]
    metadata: dict = field(default_factory=dict)
    size: int = 0
    dtypes: dict[str, str] = field(default_factory=dict)


def read_header(path: str | os.PathLike) -> Header:
    path = os.fspath(path)
    with open(path, "rb") as handle:
        magic = handle.read(9)
    size = os.path.getsize(path)
    if magic[:4] == b"GGUF":
        return _gguf(path, size)
    # safetensors first: its 8-byte length can start with any byte, including pickle's 0x80.
    if len(magic) == 9 and magic[8:9] == b"{":
        (length,) = struct.unpack("<Q", magic[:8])
        if 2 <= length <= min(MAX_HEADER, size - 8):
            return _safetensors(path, size)
    if magic[:2] == b"PK":
        return _ckpt_zip(path, size)
    if magic[:1] == b"\x80":
        return _ckpt_legacy(path, size)
    raise HeaderError(f"{os.path.basename(path)} is not a safetensors, GGUF or torch checkpoint file.")


# safetensors -------------------------------------------------------------------
def _safetensors(path: str, size: int) -> Header:
    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        raw = handle.read(length)
    try:
        data = json.loads(raw)
    except ValueError as e:
        raise HeaderError(f"Damaged safetensors header in {os.path.basename(path)}: {e}") from e
    if not isinstance(data, dict):
        raise HeaderError("safetensors header is not a JSON object.")
    metadata = data.pop("__metadata__", None) or {}
    tensors, dtypes = {}, {}
    for name, info in data.items():
        if isinstance(info, dict):
            shape = info.get("shape")
            tensors[name] = tuple(shape) if isinstance(shape, list) else None
            if info.get("dtype"):
                dtypes[name] = info["dtype"]
    return Header("safetensors", tensors, metadata if isinstance(metadata, dict) else {}, size, dtypes)


# GGUF --------------------------------------------------------------------------
_GGUF_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def _gguf(path: str, size: int) -> Header:
    with open(path, "rb") as f:
        def read(fmt):
            n = struct.calcsize(fmt)
            data = f.read(n)
            if len(data) != n:
                raise HeaderError("Truncated GGUF header.")
            return struct.unpack(fmt, data)[0]

        def string():
            n = read("<Q")
            if n > 1 << 24:
                raise HeaderError("Implausible GGUF string length.")
            return f.read(n).decode("utf-8", "replace")

        def value(vtype):
            if vtype in _GGUF_SCALARS:
                return read(_GGUF_SCALARS[vtype])
            if vtype == 8:
                return string()
            if vtype == 9:
                itype, count = read("<I"), read("<Q")
                if count > 1 << 20:
                    # Huge arrays (tokenizer vocabularies) are skipped, not kept.
                    for _ in range(count):
                        value(itype)
                    return f"<array of {count}>"
                return [value(itype) for _ in range(count)]
            raise HeaderError(f"Unknown GGUF value type {vtype}.")

        f.read(4)
        version = read("<I")
        if version < 2:
            raise HeaderError("GGUF version 1 is not supported.")
        n_tensors, n_kv = read("<Q"), read("<Q")
        if n_tensors > 1 << 20 or n_kv > 1 << 20:
            raise HeaderError("Implausible GGUF header counts.")
        metadata = {}
        for _ in range(n_kv):
            key = string()
            metadata[key] = value(read("<I"))
        tensors: dict[str, tuple[int, ...] | None] = {}
        for _ in range(n_tensors):
            name = string()
            dims = [read("<Q") for _ in range(read("<I"))]
            read("<I")  # ggml type
            read("<Q")  # offset
            tensors[name] = tuple(reversed(dims))  # GGUF stores dims innermost first; torch order is the reverse
    return Header("gguf", tensors, {k: v for k, v in metadata.items() if not isinstance(v, list) or len(v) < 64}, size)


# Pickle checkpoints (never unpickled) ------------------------------------------
class _Global:
    def __init__(self, module: str, name: str):
        self.module, self.name = module, name

    def __repr__(self):
        return f"{self.module}.{self.name}"


class _Call:
    def __init__(self, func, args) -> None:
        self.func, self.args = func, args
        self.state: dict | None = None
        self.items: dict = {}


class _Mark:
    pass


def _tensor_shape(obj):
    """The shape of a torch._utils._rebuild_tensor_v2(storage, offset, size, stride, ...) call, else None."""
    if isinstance(obj, _Call) and isinstance(obj.func, _Global) and obj.func.name.startswith("_rebuild"):
        args = obj.args if isinstance(obj.args, tuple) else ()
        for arg in args[1:4]:
            if isinstance(arg, tuple) and all(isinstance(x, int) for x in arg):
                return arg
        return ()
    if isinstance(obj, _Call) and isinstance(obj.func, _Global) and obj.func.name == "_rebuild_parameter":
        return _tensor_shape(obj.args[0]) if obj.args else None
    return None


def _walk_pickle(data: bytes) -> object:
    """Run a pickle's opcodes on a data-only stack. GLOBAL never imports; REDUCE never calls."""
    stack: list = []
    memo: dict = {}
    marks: list[int] = []

    def pop_mark():
        start = marks.pop()
        items = stack[start:]
        del stack[start:]
        return items

    result = None
    for op, arg, _pos in pickletools.genops(data):
        name = op.name
        if name in ("PROTO", "FRAME"):
            continue
        if name == "MARK":
            marks.append(len(stack))
        elif name in ("EMPTY_DICT",):
            stack.append({})
        elif name in ("EMPTY_LIST",):
            stack.append([])
        elif name == "EMPTY_TUPLE":
            stack.append(())
        elif name == "DICT":
            items = pop_mark()
            stack.append({items[i]: items[i + 1] for i in range(0, len(items) - 1, 2) if _hashable(items[i])})
        elif name == "LIST":
            stack.append(pop_mark())
        elif name == "TUPLE":
            stack.append(tuple(pop_mark()))
        elif name in ("TUPLE1", "TUPLE2", "TUPLE3"):
            n = int(name[-1])
            items = tuple(stack[-n:])
            del stack[-n:]
            stack.append(items)
        elif name in ("BINUNICODE", "SHORT_BINUNICODE", "UNICODE", "BINUNICODE8", "STRING", "BINSTRING",
                      "SHORT_BINSTRING", "BINBYTES", "SHORT_BINBYTES", "BINBYTES8"):
            stack.append(arg)
        elif name in ("BININT", "BININT1", "BININT2", "INT", "LONG", "LONG1", "LONG4", "BINFLOAT", "FLOAT"):
            stack.append(arg)
        elif name == "NONE":
            stack.append(None)
        elif name in ("NEWTRUE", "NEWFALSE"):
            stack.append(name == "NEWTRUE")
        elif name in ("GLOBAL", "INST"):
            module, _, attr = str(arg).partition(" ")
            stack.append(_Global(module, attr))
        elif name == "STACK_GLOBAL":
            attr, module = stack.pop(), stack.pop()
            stack.append(_Global(str(module), str(attr)))
        elif name in ("REDUCE", "NEWOBJ"):
            args, func = stack.pop(), stack.pop()
            stack.append(_Call(func, args))
        elif name == "NEWOBJ_EX":
            _kwargs, args, func = stack.pop(), stack.pop(), stack.pop()
            stack.append(_Call(func, args))
        elif name == "BUILD":
            state = stack.pop()
            if isinstance(stack[-1], _Call) and isinstance(state, dict):
                stack[-1].state = state
        elif name in ("BINPERSID", "PERSID"):
            stack[-1:] = [("persistent", stack[-1] if name == "BINPERSID" else arg)]
        elif name in ("BINPUT", "LONG_BINPUT", "PUT"):
            memo[arg] = stack[-1]
        elif name == "MEMOIZE":
            memo[len(memo)] = stack[-1]
        elif name in ("BINGET", "LONG_BINGET", "GET"):
            stack.append(memo.get(arg))
        elif name == "SETITEM":
            value, key = stack.pop(), stack.pop()
            if isinstance(stack[-1], dict) and _hashable(key):
                stack[-1][key] = value
        elif name == "SETITEMS":
            items = pop_mark()
            target = stack[-1]
            if isinstance(target, dict):
                for i in range(0, len(items) - 1, 2):
                    if _hashable(items[i]):
                        target[items[i]] = items[i + 1]
            elif isinstance(target, _Call):  # OrderedDict() filled by SETITEMS
                state = target.items
                for i in range(0, len(items) - 1, 2):
                    if _hashable(items[i]):
                        state[items[i]] = items[i + 1]
                target.items = state
        elif name == "APPEND":
            value = stack.pop()
            if isinstance(stack[-1], list):
                stack[-1].append(value)
        elif name == "APPENDS":
            items = pop_mark()
            if isinstance(stack[-1], list):
                stack[-1].extend(items)
        elif name == "POP":
            stack.pop()
        elif name == "POP_MARK":
            pop_mark()
        elif name == "DUP":
            stack.append(stack[-1])
        elif name == "STOP":
            result = stack[-1] if stack else None
            break
        else:
            raise HeaderError(f"Unsupported pickle opcode {name}.")
    return result


def _hashable(value) -> bool:
    try:
        hash(value)
    except TypeError:
        return False
    return True


def _state_dict(obj) -> dict:
    """The tensor dict inside a loaded checkpoint object: {'state_dict': {...}} or the dict itself."""
    if isinstance(obj, _Call):
        obj = getattr(obj, "items", None) or {}
    if not isinstance(obj, dict):
        return {}
    for key in ("state_dict", "model", "module"):
        inner = obj.get(key)
        if isinstance(inner, _Call):
            inner = getattr(inner, "items", None)
        if isinstance(inner, dict) and inner:
            return inner
    return obj


def _ckpt_tensors(obj) -> tuple[dict, dict]:
    sd = _state_dict(obj)
    tensors, extra = {}, {}
    for key, value in sd.items():
        if not isinstance(key, str):
            continue
        shape = _tensor_shape(value)
        if shape is not None:
            tensors[key] = shape
        elif isinstance(value, (str, int, float)):
            extra[key] = value
    return tensors, extra


def _ckpt_zip(path: str, size: int) -> Header:
    try:
        with zipfile.ZipFile(path) as archive:
            name = next((n for n in archive.namelist() if n.endswith("data.pkl")), None)
            if not name:
                raise HeaderError("Zip file without a data.pkl: not a torch checkpoint.")
            data = archive.read(name)
    except zipfile.BadZipFile as e:
        raise HeaderError(f"Damaged checkpoint zip: {e}") from e
    tensors, extra = _ckpt_tensors(_walk_pickle(data))
    return Header("ckpt", tensors, {k: str(v) for k, v in extra.items()}, size)


def _ckpt_legacy(path: str, size: int) -> Header:
    # Legacy torch.save: pickles in a row (magic number, protocol, sys info, the object, storage keys),
    # then raw storage bytes. Only a bounded prefix is read.
    with open(path, "rb") as handle:
        data = handle.read(min(size, 64 * 1024 * 1024))
    offset, objects = 0, []
    for _ in range(4):
        end = None
        try:
            for op, _arg, pos in pickletools.genops(data[offset:]):
                if op.name == "STOP" and pos is not None:
                    end = offset + pos + 1
                    break
        except ValueError as e:
            raise HeaderError(f"Unreadable legacy torch file: {e}") from e
        if end is None:
            break
        objects.append(_walk_pickle(data[offset:end]))
        offset = end
    if len(objects) < 4:
        raise HeaderError("Not a legacy torch checkpoint.")
    tensors, extra = _ckpt_tensors(objects[3])
    return Header("ckpt", tensors, {k: str(v) for k, v in extra.items()}, size)
