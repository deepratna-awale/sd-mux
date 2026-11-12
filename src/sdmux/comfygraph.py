"""
ComfyUI's node contract, and a graph builder that works for any installed node.

ComfyUI describes every node it has loaded, custom packs included, at /object_info: inputs (type,
default, range, combo options), outputs (types and names), category and the Python module it
came from. This module turns that into typed specs and lets you wire graphs from them:

    sd = sdmux.connect("http://127.0.0.1:8188")
    sd.nodes()                                # every node: name, category, pack
    spec = sd.node("KSampler")                # typed inputs and outputs
    g = sd.graph()
    ckpt = g.add("CheckpointLoaderSimple", ckpt_name="waiIllustrious")      # loose file-name matching
    pos = g.add("CLIPTextEncode", text="1girl", clip=ckpt.CLIP)            # outputs by name
    neg = g.add("CLIPTextEncode", text="lowres", clip=ckpt.CLIP)
    lat = g.add("EmptyLatentImage", width=1024, height=1024)              # missing inputs get server defaults
    s = g.add("KSampler", model=ckpt.MODEL, positive=pos, negative=neg, latent_image=lat, steps=28)
    img = g.add("VAEDecode", samples=s, vae=ckpt.VAE)
    g.add("PreviewImage", images=img)
    r = sd.run(g)                             # validated against /object_info first

A node with a single output can be passed directly as a link (pos above). Validation reports every
problem at once: unknown node or input, missing required input, link of the wrong type, number out
of range, value not among the combo options.
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field

from .errors import Rejected
from .names import match_choice

PRIMITIVES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}


class GraphError(Rejected):
    """The graph does not fit the server's node contract. .problems lists every issue."""

    code = "invalid_graph"

    def __init__(self, problems: list[str], **kw):
        self.problems = problems
        super().__init__("Invalid ComfyUI graph:\n  " + "\n  ".join(problems), **kw)


@dataclass
class InputSpec:
    name: str
    type: str  # INT, FLOAT, STRING, BOOLEAN, COMBO, or a link type such as MODEL, IMAGE, "*"
    required: bool
    default: object = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: list | None = None  # COMBO choices
    tooltip: str | None = None
    raw: object = None

    @property
    def is_link(self) -> bool:
        return self.type not in PRIMITIVES

    def accepts(self, link_type: str) -> bool:
        if self.type == "*" or link_type == "*":
            return True
        wanted = {t.strip() for t in self.type.split(",")}
        return link_type in wanted or (self.type in PRIMITIVES and link_type == self.type)


@dataclass
class OutputSpec:
    index: int
    name: str
    type: str
    is_list: bool = False


@dataclass
class NodeSpec:
    name: str
    display_name: str = ""
    category: str = ""
    description: str = ""
    module: str = ""  # "nodes", "comfy_extras.nodes_...", or "custom_nodes.<pack>"
    inputs: dict[str, InputSpec] = field(default_factory=dict)
    outputs: list[OutputSpec] = field(default_factory=list)
    output_node: bool = False
    deprecated: bool = False
    raw: dict = field(default_factory=dict)

    @property
    def pack(self) -> str:
        """The custom node pack this node comes from, or "core"."""
        if self.module.startswith("custom_nodes."):
            return self.module.split(".", 2)[1]
        return "core"

    @property
    def required(self) -> list[str]:
        return [n for n, i in self.inputs.items() if i.required]

    def output(self, key) -> OutputSpec:
        """An output by index or (case-insensitive) name or type."""
        if isinstance(key, int):
            return self.outputs[key]
        for attr in ("name", "type"):
            hits = [o for o in self.outputs if getattr(o, attr).lower() == str(key).lower()]
            if len(hits) == 1:
                return hits[0]
            if hits:
                raise KeyError(f"{self.name} has several outputs called {key!r}; use an index.")
        raise KeyError(f"{self.name} has no output {key!r}. Outputs: {', '.join(o.name for o in self.outputs) or 'none'}")

    def __str__(self) -> str:
        lines = [f"{self.name} ({self.category}, {self.pack})"]
        for i in self.inputs.values():
            extra = []
            if i.default is not None:
                extra.append(f"default {i.default!r}")
            if i.min is not None or i.max is not None:
                extra.append(f"range {i.min}..{i.max}")
            if i.options is not None:
                extra.append(f"{len(i.options)} options")
            lines.append(f"  in  {i.name}: {i.type}{'' if i.required else ' (optional)'}{'  ' + ', '.join(extra) if extra else ''}")
        for o in self.outputs:
            lines.append(f"  out {o.index} {o.name}: {o.type}")
        return "\n".join(lines)


def parse_spec(name: str, info: dict) -> NodeSpec:
    """A NodeSpec from one /object_info entry (old list-combo and new COMBO formats)."""
    spec = NodeSpec(
        name=name,
        display_name=info.get("display_name") or name,
        category=info.get("category") or "",
        description=info.get("description") or "",
        module=info.get("python_module") or "",
        output_node=bool(info.get("output_node")),
        deprecated=bool(info.get("deprecated")),
        raw=info,
    )
    for section, required in (("required", True), ("optional", False)):
        for input_name, entry in (info.get("input", {}).get(section) or {}).items():
            spec.inputs[input_name] = _parse_input(input_name, entry, required)
    types = info.get("output") or []
    names = info.get("output_name") or types
    lists = info.get("output_is_list") or [False] * len(types)
    for i, t in enumerate(types):
        spec.outputs.append(OutputSpec(i, str(names[i] if i < len(names) else t), str(t), bool(lists[i] if i < len(lists) else False)))
    return spec


def _parse_input(name: str, entry, required: bool) -> InputSpec:
    entry = entry if isinstance(entry, (list, tuple)) else [entry]
    first = entry[0] if entry else "*"
    opts = entry[1] if len(entry) > 1 and isinstance(entry[1], dict) else {}
    if isinstance(first, list):
        kind, options = "COMBO", list(first)
    elif first == "COMBO":
        kind, options = "COMBO", list(opts.get("options") or [])
    else:
        kind, options = str(first), None
    default = opts.get("default")
    if default is None and options:
        default = options[0]
    return InputSpec(name, kind, required, default, opts.get("min"), opts.get("max"), opts.get("step"), options,
                     opts.get("tooltip"), entry)


# Graph -----------------------------------------------------------------------------
class Out:
    """A link to one output of a node: serialises to ComfyUI's [node_id, index]."""

    def __init__(self, node: "NodeRef", index: int, type_: str, name: str):
        self.node, self.index, self.type, self.name = node, index, type_, name

    def to_api(self) -> list:
        return [self.node.id, self.index]

    def __repr__(self) -> str:
        return f"<Out {self.node.id}.{self.name}:{self.type}>"


class NodeRef:
    """A node in a Graph. Outputs: ref.MODEL, ref["MODEL"], ref[0], or the ref itself when there is one output."""

    def __init__(self, graph: "Graph", node_id: str):
        self.graph = graph
        self.id = node_id

    @property
    def class_type(self) -> str:
        return self.graph.nodes[self.id]["class_type"]

    @property
    def inputs(self) -> dict:
        return self.graph.nodes[self.id]["inputs"]

    @property
    def spec(self) -> NodeSpec | None:
        return self.graph.spec(self.class_type)

    def out(self, key) -> Out:
        spec = self.spec
        if spec is None:
            if isinstance(key, int):
                return Out(self, key, "*", str(key))
            raise KeyError(f"Unknown node {self.class_type}; refer to its outputs by index.")
        o = spec.output(key)
        return Out(self, o.index, o.type, o.name)

    def __getattr__(self, key) -> Out:
        if key.startswith("_"):
            raise AttributeError(key)
        try:
            return self.out(key)
        except KeyError as e:
            raise AttributeError(str(e)) from None

    def __getitem__(self, key) -> Out:
        return self.out(key)

    def set(self, **inputs) -> "NodeRef":
        """Change inputs of this node (values or links)."""
        self.graph._set_inputs(self.id, inputs)
        return self

    def __repr__(self) -> str:
        return f"<Node {self.id} {self.class_type}>"


class Graph:
    """
    An API-format ComfyUI workflow built against a server's node contract.

    specs is a callable name -> NodeSpec | None (ComfyUI.node), so validation and defaults follow the
    server you will run on. A Graph can also wrap an existing API-format dict (Graph.from_api).
    """

    def __init__(self, specs=None, fill_defaults: bool = True, match_files: bool = True):
        self._specs = specs
        self.fill_defaults = fill_defaults
        self.match_files = match_files
        self.nodes: dict[str, dict] = {}
        self._ids = itertools.count(1)

    @classmethod
    def from_api(cls, workflow: dict, specs=None) -> "Graph":
        g = cls(specs, fill_defaults=False)
        g.nodes = copy.deepcopy(workflow)
        numeric = [int(k) for k in workflow if str(k).isdigit()]
        g._ids = itertools.count(max(numeric, default=0) + 1)
        return g

    def spec(self, class_type: str) -> NodeSpec | None:
        return self._specs(class_type) if self._specs else None

    def _new_id(self) -> str:
        while True:
            node_id = str(next(self._ids))
            if node_id not in self.nodes:
                return node_id

    def add(self, class_type: str, _id: str | None = None, **inputs) -> NodeRef:
        """Add a node. Inputs are values, Out links, or NodeRefs (for single-output nodes)."""
        node_id = str(_id) if _id is not None else self._new_id()
        if node_id in self.nodes:
            raise ValueError(f"Node id {node_id!r} is already used.")
        self.nodes[node_id] = {"class_type": class_type, "inputs": {}}
        spec = self.spec(class_type)
        if spec and self.fill_defaults:
            for name, i in spec.inputs.items():
                if i.required and not i.is_link and name not in inputs:
                    default = "" if i.default is None and i.type == "STRING" else i.default  # text widgets start empty
                    if default is not None:
                        self.nodes[node_id]["inputs"][name] = default
        self._set_inputs(node_id, inputs)
        return NodeRef(self, node_id)

    def ref(self, node_id: str) -> NodeRef:
        if str(node_id) not in self.nodes:
            raise KeyError(f"No node {node_id!r} in this graph.")
        return NodeRef(self, str(node_id))

    def find(self, class_type: str) -> list[NodeRef]:
        return [NodeRef(self, k) for k, v in self.nodes.items() if v["class_type"] == class_type]

    def remove(self, node_id: str) -> None:
        del self.nodes[str(node_id)]

    def _set_inputs(self, node_id: str, inputs: dict) -> None:
        node = self.nodes[node_id]
        spec = self.spec(node["class_type"])
        for name, value in inputs.items():
            if isinstance(value, NodeRef):
                outs = value.spec.outputs if value.spec else []
                if len(outs) != 1:
                    raise ValueError(f"{value!r} has {len(outs)} outputs; pick one, e.g. node.{outs[0].name if outs else 'OUTPUT'}.")
                value = value.out(0)
            if spec and self.match_files and isinstance(value, str) and name in spec.inputs:
                i = spec.inputs[name]
                if i.options and value not in i.options:
                    value = match_choice(value, [str(o) for o in i.options]) or value
            node["inputs"][name] = value

    def to_api(self) -> dict:
        """The API-format dict ComfyUI's /prompt takes."""
        out = {}
        for node_id, node in self.nodes.items():
            inputs = {k: (v.to_api() if isinstance(v, Out) else v) for k, v in node["inputs"].items()}
            out[node_id] = {"class_type": node["class_type"], "inputs": inputs}
        return out

    def validate(self) -> None:
        """Check every node against the node contract; raise GraphError listing all problems."""
        problems: list[str] = []
        api = self.to_api()
        for node_id, node in api.items():
            ct = node["class_type"]
            spec = self.spec(ct)
            where = f"node {node_id} ({ct})"
            if spec is None:
                problems.append(f"{where}: ComfyUI has no node {ct!r} (custom node pack not installed?)")
                continue
            for name, value in node["inputs"].items():
                i = spec.inputs.get(name)
                if i is None:
                    problems.append(f"{where}: no input {name!r}. Inputs: {', '.join(spec.inputs)}")
                    continue
                if _is_link(value):
                    source = api.get(str(value[0]))
                    if source is None:
                        problems.append(f"{where}.{name}: links to missing node {value[0]!r}")
                        continue
                    src_spec = self.spec(source["class_type"])
                    if src_spec is None:
                        continue  # reported on that node
                    if not 0 <= value[1] < len(src_spec.outputs):
                        problems.append(f"{where}.{name}: {source['class_type']} has no output {value[1]}")
                        continue
                    link_type = src_spec.outputs[value[1]].type
                    if not i.accepts(link_type):
                        problems.append(f"{where}.{name}: expects {i.type}, got {link_type} from node {value[0]} ({source['class_type']})")
                    continue
                if i.is_link and i.type != "*":
                    problems.append(f"{where}.{name}: expects a {i.type} link, got a value {value!r}")
                    continue
                problems.extend(_check_value(where, i, value))
            for name in spec.required:
                i = spec.inputs[name]
                if name not in node["inputs"]:
                    problems.append(f"{where}: missing required input {name!r} ({i.type})")
        if not any((self.spec(n["class_type"]) or NodeSpec("")).output_node for n in api.values()):
            problems.append("graph has no output node (add PreviewImage or SaveImage)")
        if problems:
            raise GraphError(problems, backend="comfyui")

    def __len__(self) -> int:
        return len(self.nodes)

    def __repr__(self) -> str:
        return f"<Graph {len(self.nodes)} nodes>"


def _is_link(value) -> bool:
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], (str, int)) and isinstance(value[1], int)


def _check_value(where: str, i: InputSpec, value) -> list[str]:
    name = f"{where}.{i.name}"
    if i.type == "COMBO":
        if i.options and value not in i.options:
            shown = ", ".join(map(str, i.options[:8])) + (" ..." if len(i.options) > 8 else "")
            return [f"{name}: {value!r} is not one of the options ({shown})"]
        return []
    if i.type in ("INT", "FLOAT"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return [f"{name}: expects {i.type}, got {value!r}"]
        if i.type == "INT" and isinstance(value, float) and not value.is_integer():
            return [f"{name}: expects an integer, got {value!r}"]
        if i.min is not None and value < i.min:
            return [f"{name}: {value} is below the minimum {i.min}"]
        if i.max is not None and value > i.max:
            return [f"{name}: {value} is above the maximum {i.max}"]
        return []
    if i.type == "STRING" and not isinstance(value, str):
        return [f"{name}: expects STRING, got {value!r}"]
    if i.type == "BOOLEAN" and not isinstance(value, bool):
        return [f"{name}: expects BOOLEAN, got {value!r}"]
    return []
