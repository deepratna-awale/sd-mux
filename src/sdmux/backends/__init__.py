from .a1111 import A1111, Forge, SDNext
from .base import Backend
from .comfyui import ComfyUI
from .drawthings import DrawThings
from .invokeai import InvokeAI
from .sdcpp import SDCpp
from .swarmui import SwarmUI

BACKENDS: dict[str, type[Backend]] = {
    "a1111": A1111,
    "forge": Forge,
    "reforge": Forge,
    "sdnext": SDNext,
    "comfyui": ComfyUI,
    "swarmui": SwarmUI,
    "invokeai": InvokeAI,
    "drawthings": DrawThings,
    "sdcpp": SDCpp,
}

__all__ = ["Backend", "A1111", "Forge", "SDNext", "ComfyUI", "SwarmUI", "InvokeAI", "DrawThings", "SDCpp", "BACKENDS"]
