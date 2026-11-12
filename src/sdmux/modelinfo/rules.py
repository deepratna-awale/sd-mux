"""
Architecture, kind, variant and prediction type from tensor names and shapes.

The signatures are facts about each model family's published layout (the same facts
ComfyUI's model detection relies on); this table and its code are sdmux's own.
Each rule records why it fired, so callers can see the evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

Shapes = dict  # tensor name -> shape tuple or None

UNET_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "model.model.", "net.", "model.", "unet.", "transformer.")

# Cross-attention context width -> architecture (UNet models).
CONTEXT_WIDTH = {768: "sd15", 1024: "sd2", 2048: "sdxl", 1280: "sdxl-refiner"}


@dataclass
class Finding:
    value: str | None
    rule: str
    confidence: str = "high"


def split_prefix(tensors: Shapes) -> tuple[str, Shapes]:
    """Find the diffusion model's key prefix (the most common candidate) and return the keys under it, prefix stripped."""
    counts = {p: 0 for p in UNET_PREFIXES}
    for key in tensors:
        for prefix in UNET_PREFIXES:
            if key.startswith(prefix):
                counts[prefix] += 1
                break
    best = max(counts, key=counts.get)
    if counts[best] > 5 or (counts[best] >= 2 and counts[best] * 2 >= len(tensors)):
        return best, {k[len(best):]: v for k, v in tensors.items() if k.startswith(best)}
    return "", dict(tensors)


def _has(keys, *names) -> bool:
    return all(n in keys for n in names)


def _any_suffix(keys, stem: str, suffixes=("weight", "scale")) -> bool:
    return any(stem + s in keys for s in suffixes)


def _shape(tensors: Shapes, key: str, dim: int):
    shape = tensors.get(key)
    if shape and len(shape) > dim:
        return shape[dim]
    return None


# Transformer / DiT families: (architecture, rule name, test on the prefix-stripped keys)
def _dit_rules(u: Shapes) -> tuple[str, str] | None:
    k = u.keys()
    if "joint_blocks.0.context_block.attn.qkv.weight" in k:
        return "sd3", "joint_blocks.0.context_block.attn.qkv.weight (MMDiT)"
    if "clf.1.weight" in k:
        return "stable-cascade", "clf.1.weight"
    if "double_layers.0.attn.w1q.weight" in k:
        return "auraflow", "double_layers.0.attn.w1q.weight"
    if "mlp_t5.0.weight" in k:
        return "hunyuan-dit", "mlp_t5.0.weight"
    if "txt_in.individual_token_refiner.blocks.0.norm1.weight" in k:
        return "hunyuan-video", "txt_in.individual_token_refiner.blocks.0.norm1.weight"
    if _any_suffix(k, "double_blocks.0.img_attn.norm.key_norm.") and (
        "img_in.weight" in k or _any_suffix(k, "distilled_guidance_layer.norms.0.") or _any_suffix(k, "distilled_guidance_layer.0.norms.0.")
    ):
        if "double_stream_modulation_img.lin.weight" in k:
            return "flux2", "double_blocks + double_stream_modulation_img.lin.weight"
        if _any_suffix(k, "distilled_guidance_layer.norms.0.") or _any_suffix(k, "distilled_guidance_layer.0.norms.0."):
            if _any_suffix(k, "nerf_blocks.0.norm."):
                return "chroma-radiance", "flux layout + distilled_guidance_layer + nerf_blocks"
            return "chroma", "flux layout + distilled_guidance_layer"
        return "flux", "double_blocks.0.img_attn.norm.key_norm + img_in.weight"
    if "t5_yproj.weight" in k:
        return "mochi", "t5_yproj.weight"
    if _has(k, "adaln_single.emb.timestep_embedder.linear_1.bias", "pos_embed.proj.bias"):
        return "pixart", "adaln_single + pos_embed.proj.bias (diffusers PixArt)"
    if "adaln_single.emb.timestep_embedder.linear_1.bias" in k:
        return "ltxv", "adaln_single.emb.timestep_embedder.linear_1.bias"
    if "t_block.1.weight" in k:
        return "pixart", "t_block.1.weight"
    if "blocks.block0.blocks.0.block.attn.to_q.0.weight" in k or "blocks.0.mlp.layer1.weight" in k:
        return "cosmos", "cosmos block layout"
    if _has(k, "cap_embedder.1.weight", "noise_refiner.0.attention.k_norm.weight"):
        return "lumina2", "cap_embedder.1.weight + noise_refiner.0.attention.k_norm.weight (Lumina 2 / Z-Image)"
    if "blocks.0.norm1.linear.weight" in k:
        return "cogvideox", "blocks.0.norm1.linear.weight"
    if "head.modulation" in k:
        return "wan", "head.modulation"
    if "caption_projection.0.linear.weight" in k:
        return "hidream", "caption_projection.0.linear.weight"
    if "txt_norm.weight" in k:
        return "qwen-image", "txt_norm.weight"
    # Diffusers-format Flux transformer
    if _has(k, "single_transformer_blocks.0.attn.norm_k.weight", "x_embedder.weight", "context_embedder.weight") or (
        "transformer_blocks.0.attn.norm_added_k.weight" in k and "single_transformer_blocks.0.norm.linear.weight" in k
    ):
        return "flux", "diffusers Flux transformer (single_transformer_blocks + x_embedder)"
    return None


def _unet_arch(u: Shapes, full: Shapes) -> Finding | None:
    """SD 1.x / 2.x / SDXL / refiner by cross-attention width; key-only fallbacks for shape-less (pickle) headers."""
    k = u.keys()
    unet = "input_blocks.0.0.weight" in k or "conv_in.weight" in k and any(".attn2." in x for x in k)
    if not unet:
        return None
    widths = {}
    for key, shape in u.items():
        if key.endswith("attn2.to_k.weight") and shape and len(shape) == 2:
            widths[shape[1]] = widths.get(shape[1], 0) + 1
    if widths:
        width = max(widths, key=widths.get)
        arch = CONTEXT_WIDTH.get(width)
        if arch:
            return Finding(arch, f"attn2.to_k.weight context width {width}")
        return Finding(None, f"UNet with unknown context width {width}", "low")
    label = _shape(u, "label_emb.0.0.weight", 1)
    if label == 2816:
        return Finding("sdxl", "label_emb.0.0.weight width 2816")
    if label == 2560:
        return Finding("sdxl-refiner", "label_emb.0.0.weight width 2560")
    fk = full.keys()
    if any(x.startswith("conditioner.embedders.1.") for x in fk) or "label_emb.0.0.weight" in k:
        return Finding("sdxl", "conditioner.embedders.1 / label_emb keys", "medium")
    if any(x.startswith("cond_stage_model.model.") for x in fk):
        return Finding("sd2", "cond_stage_model.model (OpenCLIP) keys", "medium")
    if any(x.startswith("cond_stage_model.transformer.") for x in fk):
        return Finding("sd15", "cond_stage_model.transformer (CLIP-L) keys", "medium")
    return Finding("sd15", "UNet layout without attention shapes", "low")


# Kinds ------------------------------------------------------------------------
LORA_MARKERS = (".lora_down.", ".lora_up.", ".lora_A.", ".lora_B.", ".lokr_w1", ".hada_w1_a", ".oft_blocks", ".alpha",
                "lora_unet_", "lora_te_", "lora_te1_", "lora_te2_", "lora_te3_", ".lora.down.", ".lora.up.", ".dora_scale")


def is_lora(tensors: Shapes) -> bool:
    hits = sum(1 for key in tensors if any(m in key for m in LORA_MARKERS[:9]) or key.startswith(("lora_unet_", "lora_te")))
    return hits > 0 and hits >= len(tensors) * 0.3


def is_vae(tensors: Shapes) -> bool:
    keys = tensors.keys()
    enc = any(k.startswith(("encoder.down.", "encoder.down_blocks.", "first_stage_model.encoder.", "encoder.conv_in")) for k in keys)
    dec = any(k.startswith(("decoder.up.", "decoder.up_blocks.", "first_stage_model.decoder.", "decoder.conv_in")) for k in keys)
    unet = any(".attn2." in k or "double_blocks." in k or "joint_blocks." in k for k in keys)
    return (enc or dec) and not unet and len(tensors) < 600


def is_text_encoder(tensors: Shapes) -> str | None:
    keys = list(tensors)
    if not keys or any(".attn2." in k or "input_blocks." in k or "double_blocks." in k for k in keys):
        return None
    joined = keys[:400]
    if any(k.startswith(("text_model.encoder.", "transformer.text_model.", "text_model.embeddings.")) for k in joined):
        width = _shape(tensors, "text_model.final_layer_norm.weight", 0) or _shape(tensors, "transformer.text_model.final_layer_norm.weight", 0)
        return {768: "clip-l", 1024: "clip-h", 1280: "clip-g"}.get(width, "clip")
    if any(k.startswith(("encoder.block.0.", "shared.weight")) for k in joined) and any("SelfAttention" in k for k in joined):
        return "t5"
    if any(k.startswith(("model.layers.0.", "layers.0.", "model.language_model.layers.0.")) for k in joined) and any(
        "self_attn" in k for k in joined
    ):
        return "llm"
    return None


def is_controlnet(tensors: Shapes) -> bool:
    keys = tensors.keys()
    return any(
        k.startswith(("control_model.", "input_hint_block.", "controlnet_cond_embedding.", "zero_convs.", "controlnet_blocks.",
                      "controlnet_x_embedder", "lllite_"))
        or ".input_hint_block." in k
        for k in keys
    )


def is_embedding(tensors: Shapes) -> bool:
    keys = set(tensors)
    return bool(keys) and len(keys) <= 4 and bool(keys & {"emb_params", "string_to_param.*", "clip_l", "clip_g", "<concept>"} or
                                                       any(k.startswith("string_to_param") for k in keys))


# LoRA architecture -------------------------------------------------------------
def lora_architecture(tensors: Shapes) -> Finding | None:
    keys = list(tensors)
    if any(k.startswith(("lora_te1_", "lora_te2_")) for k in keys):
        return Finding("sdxl", "lora_te1_/lora_te2_ keys (two text encoders)")
    if any("double_blocks" in k or "single_blocks" in k or "single_transformer_blocks" in k for k in keys):
        if any("double_stream_modulation" in k for k in keys):
            return Finding("flux2", "LoRA on double_stream_modulation layers")
        return Finding("flux", "LoRA on double_blocks/single_blocks")
    if any("joint_blocks" in k for k in keys):
        return Finding("sd3", "LoRA on joint_blocks")
    if any("transformer_blocks" in k and ("img_mlp" in k or "txt_mlp" in k) for k in keys):
        return Finding("qwen-image", "LoRA on img_mlp/txt_mlp")
    widths = {}
    for key, shape in tensors.items():
        norm = key.replace("_", ".")
        if "attn2" in norm and "to.k" in norm and ("lora.down" in norm or "lora.A" in norm or "lora_A" in key) and shape and len(shape) == 2:
            widths[shape[1]] = widths.get(shape[1], 0) + 1
    if widths:
        width = max(widths, key=widths.get)
        if width in CONTEXT_WIDTH:
            return Finding(CONTEXT_WIDTH[width], f"LoRA attn2 to_k input width {width}")
    if any(k.startswith("lora_te_") for k in keys) or any(k.startswith("lora_unet_") for k in keys):
        if any("lora_unet_output_blocks" in k or "lora_unet_down_blocks" in k or "lora_unet_up_blocks" in k for k in keys):
            return Finding("sd15", "kohya UNet LoRA without SDXL text encoders", "medium")
    return None


# Variant / prediction ---------------------------------------------------------
def unet_variant(u: Shapes) -> Finding | None:
    channels = _shape(u, "input_blocks.0.0.weight", 1) or _shape(u, "conv_in.weight", 1)
    if channels == 9:
        return Finding("inpaint", "input_blocks.0.0.weight has 9 input channels")
    if channels == 8:
        return Finding("instruct", "input_blocks.0.0.weight has 8 input channels (instruct-pix2pix)")
    return None


def flux_variant(u: Shapes) -> Finding | None:
    channels = _shape(u, "img_in.weight", 1)
    if channels == 384:
        return Finding("fill", "img_in.weight has 384 input channels (Flux Fill)")
    if "guidance_in.in_layer.weight" in u:
        return Finding("dev", "guidance_in.in_layer.weight present (guidance-distilled dev)")
    if "img_in.weight" in u:
        return Finding("schnell", "no guidance_in layer (schnell)")
    return None


FLOW_ARCHITECTURES = {"sd3", "flux", "flux2", "chroma", "chroma-radiance", "auraflow", "qwen-image", "lumina2", "hidream", "wan",
                      "hunyuan-video", "ltxv", "cosmos", "mochi"}


def prediction(tensors: Shapes, architecture: str | None, metadata: dict) -> Finding | None:
    keys = tensors.keys()
    if "edm_vpred.sigma_max" in keys:
        return Finding("v", "edm_vpred.sigma_max key (EDM v-prediction)")
    if "edm_mean" in keys and "edm_std" in keys:
        return Finding("edm", "edm_mean + edm_std keys (Playground 2.5)")
    if "v_pred" in keys:
        return Finding("v", "v_pred key" + (" + ztsnr (zero terminal SNR)" if "ztsnr" in keys else ""))
    pt = str(metadata.get("modelspec.prediction_type", "")).lower()
    if pt:
        if pt.startswith("v"):
            return Finding("v", f"modelspec.prediction_type={pt}", "medium")
        if pt.startswith("eps") or pt == "epsilon":
            return Finding("epsilon", f"modelspec.prediction_type={pt}", "medium")
        if "flow" in pt or "rectified" in pt:
            return Finding("flow", f"modelspec.prediction_type={pt}", "medium")
    if architecture in FLOW_ARCHITECTURES:
        return Finding("flow", f"{architecture} is a flow-matching architecture")
    if architecture in ("sd15", "sdxl", "sdxl-refiner"):
        return Finding("epsilon", f"{architecture} default (no v_pred marker)", "medium")
    if architecture == "sd2":
        return Finding(None, "SD 2.x 512 models are epsilon, 768 models v-prediction; the header does not say", "low")
    return None


def ztsnr(tensors: Shapes) -> bool:
    return "ztsnr" in tensors


# Entry point --------------------------------------------------------------------
@dataclass
class Structure:
    kind: str
    architecture: Finding | None
    variant: Finding | None
    subtype: str | None = None


def analyse(tensors: Shapes, metadata: dict | None = None) -> Structure:
    """kind + architecture + variant from a header's tensors."""
    metadata = metadata or {}
    if not tensors:
        return Structure("unknown", None, None)
    if is_lora(tensors):
        return Structure("lora", lora_architecture(tensors), None)
    if is_embedding(tensors):
        width = None
        for shape in tensors.values():
            if shape and len(shape) == 2:
                width = shape[1]
        arch = {768: "sd15", 1024: "sd2"}.get(width) if width else None
        if {"clip_l", "clip_g"} <= set(tensors):
            arch = "sdxl"
        return Structure("embedding", Finding(arch, f"embedding width {width}", "medium") if arch else None, None)
    if is_controlnet(tensors):
        u = {k.split("control_model.", 1)[-1]: v for k, v in tensors.items()}
        arch = _unet_arch(u, tensors) if any(".attn2." in k for k in u) else None
        return Structure("controlnet", arch, None)
    if is_vae(tensors):
        channels = _shape(tensors, "decoder.conv_in.weight", 1) or _shape(tensors, "first_stage_model.decoder.conv_in.weight", 1)
        arch = {4: "sd", 16: "flux-vae", 32: "flux2-vae"}.get(channels)
        return Structure("vae", Finding(arch, f"decoder.conv_in has {channels} latent channels", "medium") if arch else None, None)
    encoder = is_text_encoder(tensors)
    if encoder:
        return Structure("text_encoder", None, None, encoder)

    prefix, u = split_prefix(tensors)
    dit = _dit_rules(u)
    if dit:
        arch, rule = dit
        variant = flux_variant(u) if arch == "flux" else None
        return Structure(_checkpoint_or_model(tensors, prefix), Finding(arch, rule), variant)
    unet = _unet_arch(u, tensors)
    if unet:
        return Structure(_checkpoint_or_model(tensors, prefix), unet, unet_variant(u))
    return Structure("unknown", None, None)


def _checkpoint_or_model(tensors: Shapes, prefix: str) -> str:
    """A full checkpoint bundles text encoder and/or VAE; a diffusion_model file holds only the denoiser."""
    keys = tensors.keys()
    has_vae = any(k.startswith(("first_stage_model.", "vae.")) for k in keys)
    has_te = any(k.startswith(("cond_stage_model.", "conditioner.", "text_encoders.", "text_encoder.")) for k in keys)
    if has_vae or has_te:
        return "checkpoint"
    if prefix.startswith("model.diffusion_model.") and not (has_vae or has_te):
        return "diffusion_model"
    return "diffusion_model"

