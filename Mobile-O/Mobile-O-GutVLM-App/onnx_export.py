#!/usr/bin/env python3
"""
Export the *understanding half* of the finetuned Mobile-O (Gut-VLM) checkpoint to
ONNX so it runs on any laptop (Windows / Linux / Intel-Mac) via ONNX Runtime --
no Apple Silicon, no CoreML, no MLX required.

This is the cross-platform sibling of export.py (which targets CoreML + MLX for
macOS). It mirrors the exact same understanding forward pass used by
Mobile-O/inference.py and mlx_infer.py:

    image --(vision encoder + mm_projector)--> image_embeds [1,256,896]
    text  --(embed_tokens)-------------------> text_embeds  [1,n,896]
    splice [text_before, image_embeds, text_after] at the <image> sentinel
    autoregressive greedy decode through the Qwen2 decoder (with KV cache)

Because the splice happens between two spans, embed_tokens has to be callable on
its own from the host, so we emit THREE ONNX graphs (the CoreML/MLX build got
away with two because MLX exposed embed_tokens as a Python-callable module):

  1. vision_encoder.onnx  pixel_values[1,3,1024,1024] -> image_embeds[1,256,896]
                          (vision tower + mm_projector folded together, FP32)
  2. embed_tokens.onnx     input_ids[1,n]  -> text_embeds[1,n,896]   (INT8)
  3. decoder.onnx          inputs_embeds + position_ids + past_kv
                             -> logits + present_kv                  (INT8)

Quantization (default): FP32 vision + INT8 (dynamic, weight-only) LLM. The vision
tower stays FP32 because hallucination detection is precision-sensitive (same
argument export.py makes for defaulting the MLX LLM to 8-bit rather than 4-bit).

The decoder is a small hand-written Qwen2 re-implementation with an explicit,
ONNX-friendly KV cache -- NOT the HuggingFace module. transformers 5.x's Cache
API does not trace cleanly, so we rebuild the (textbook) Qwen2 math ourselves and
NUMERICALLY VERIFY it against the real HF forward before trusting it (see
--verify, on by default).

Usage:
    python onnx_export.py /path/to/checkpoints/vlm_gutvlm_hal/epoch_4
    python onnx_export.py <hf-repo-id>                    # or a HuggingFace id
    python onnx_export.py <ckpt> --output-dir onnx_models
    python onnx_export.py <ckpt> --no-quantize            # FP32 everything
    python onnx_export.py <ckpt> --quantize-vision        # INT8 vision too
"""

import argparse
import json
import math
import os
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

sys.path.insert(0, str(Path(__file__).parent))

VISION_IMAGE_RES = 1024
NUM_IMAGE_TOKENS = 256
OPSET = 17
DEFAULT_MODEL = "checkpoints/vlm_gutvlm_hal/epoch_4"

TOKENIZER_FILES = [
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "merges.txt", "vocab.json", "chat_template.jinja",
    "generation_config.json",
]


# --------------------------------------------------------------------------- #
#  Vision encoder + projector (folded)                                        #
# --------------------------------------------------------------------------- #

class VisionProjWrapper(nn.Module):
    """Vision tower -> mm_projector, folded into a single graph so the host only
    deals with already-projected image embeds [1, 256, hidden]."""

    def __init__(self, model):
        super().__init__()
        self.vision_tower = model.get_model().get_vision_tower()
        self.mm_projector = model.get_model().mm_projector

    def forward(self, images):
        feats = self.vision_tower(images)                 # [1, 256, 3072]
        return self.mm_projector(feats)                   # [1, 256, 896]


class EmbedTokens(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.embed_tokens = model.get_model().embed_tokens

    def forward(self, input_ids):
        return self.embed_tokens(input_ids)


# --------------------------------------------------------------------------- #
#  Standalone, ONNX-friendly Qwen2 decoder (verified against HF below)         #
# --------------------------------------------------------------------------- #

class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x.to(dt))


def rotate_half(x):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


class Qwen2Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg["num_attention_heads"]
        self.n_kv = cfg["num_key_value_heads"]
        self.head_dim = cfg["hidden_size"] // self.n_heads
        self.n_rep = self.n_heads // self.n_kv
        h = cfg["hidden_size"]
        self.q_proj = nn.Linear(h, self.n_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(h, self.n_kv * self.head_dim, bias=True)
        self.v_proj = nn.Linear(h, self.n_kv * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, h, bias=False)

    def forward(self, x, cos, sin, past_k, past_v, mask):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_kv, self.head_dim).transpose(1, 2)

        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin

        k = torch.cat((past_k, k), dim=2)     # [B, n_kv, T, D]
        v = torch.cat((past_v, v), dim=2)
        present_k, present_v = k, v

        # repeat kv heads to match query heads (GQA)
        k = k[:, :, None].expand(B, self.n_kv, self.n_rep, k.shape[2], self.head_dim)
        k = k.reshape(B, self.n_heads, k.shape[3], self.head_dim)
        v = v[:, :, None].expand(B, self.n_kv, self.n_rep, v.shape[2], self.head_dim)
        v = v.reshape(B, self.n_heads, v.shape[3], self.head_dim)

        attn = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn = attn + mask                    # [B,1,L,T] additive
        attn = torch.softmax(attn.float(), dim=-1).to(q.dtype)
        out = torch.matmul(attn, v)           # [B, n_heads, L, D]
        out = out.transpose(1, 2).reshape(B, L, self.n_heads * self.head_dim)
        return self.o_proj(out), present_k, present_v


class Qwen2MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h, i = cfg["hidden_size"], cfg["intermediate_size"]
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2Layer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        eps = cfg["rms_norm_eps"]
        self.input_layernorm = RMSNorm(cfg["hidden_size"], eps)
        self.self_attn = Qwen2Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg["hidden_size"], eps)
        self.mlp = Qwen2MLP(cfg)

    def forward(self, x, cos, sin, past_k, past_v, mask):
        h, pk, pv = self.self_attn(self.input_layernorm(x), cos, sin, past_k, past_v, mask)
        x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, pk, pv


class Qwen2Decoder(nn.Module):
    """Qwen2 decoder + lm_head, taking inputs_embeds and an explicit flat KV cache.

    forward(inputs_embeds, position_ids, *past) where past is
    [k0, v0, k1, v1, ...] each [B, n_kv, past_len, head_dim].
    Returns (logits, present_k0, present_v0, ...).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_layers = cfg["num_hidden_layers"]
        self.head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]
        self.layers = nn.ModuleList([Qwen2Layer(cfg) for _ in range(self.n_layers)])
        self.norm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.lm_head = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)
        theta = cfg["rope_theta"]
        inv_freq = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, inputs_embeds, position_ids, *past):
        B, L, _ = inputs_embeds.shape
        past_len = past[0].shape[2]
        total = past_len + L

        # rotary embeddings for the current query positions
        pos = position_ids.float()                                  # [B, L]
        freqs = pos[:, :, None] * self.inv_freq[None, None, :]      # [B, L, D/2]
        emb = torch.cat((freqs, freqs), dim=-1)                     # [B, L, D]
        cos = emb.cos()[:, None].to(inputs_embeds.dtype)            # [B,1,L,D]
        sin = emb.sin()[:, None].to(inputs_embeds.dtype)

        # causal mask: query abs pos = position_ids, key pos = arange(total)
        key_pos = torch.arange(total, device=inputs_embeds.device)  # [T]
        allowed = key_pos[None, None, None, :] <= position_ids[:, None, :, None]
        mask = torch.where(
            allowed,
            torch.zeros((), dtype=inputs_embeds.dtype, device=inputs_embeds.device),
            torch.full((), float("-inf"), dtype=inputs_embeds.dtype, device=inputs_embeds.device),
        )                                                           # [B,1,L,T]

        x = inputs_embeds
        presents = []
        for i, layer in enumerate(self.layers):
            x, pk, pv = layer(x, cos, sin, past[2 * i], past[2 * i + 1], mask)
            presents.extend([pk, pv])

        x = self.norm(x)
        logits = self.lm_head(x)
        return (logits, *presents)


def load_decoder_weights(decoder, sd):
    """Copy weights from the HF checkpoint state_dict into the standalone decoder."""
    m = {}
    for i in range(decoder.n_layers):
        p = f"model.layers.{i}."
        d = f"layers.{i}."
        for a in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            m[d + f"self_attn.{a}.weight"] = sd[p + f"self_attn.{a}.weight"]
        for a in ["q_proj", "k_proj", "v_proj"]:
            m[d + f"self_attn.{a}.bias"] = sd[p + f"self_attn.{a}.bias"]
        for a in ["gate_proj", "up_proj", "down_proj"]:
            m[d + f"mlp.{a}.weight"] = sd[p + f"mlp.{a}.weight"]
        m[d + "input_layernorm.weight"] = sd[p + "input_layernorm.weight"]
        m[d + "post_attention_layernorm.weight"] = sd[p + "post_attention_layernorm.weight"]
    m["norm.weight"] = sd["model.norm.weight"]
    m["lm_head.weight"] = sd["lm_head.weight"]
    missing, unexpected = decoder.load_state_dict(m, strict=False)
    # inv_freq is a non-persistent buffer -> expected "missing"; nothing else should be.
    real_missing = [k for k in missing if "inv_freq" not in k]
    assert not real_missing, f"missing decoder weights: {real_missing}"
    assert not unexpected, f"unexpected decoder weights: {unexpected}"


# --------------------------------------------------------------------------- #
#  Export routines                                                            #
# --------------------------------------------------------------------------- #

def export_vision(model, out: Path):
    print("\n--- Vision encoder + projector (ONNX FP32) ---")
    vt = model.get_model().get_vision_tower()
    if not vt.is_loaded:
        vt.load_model()
    wrapper = VisionProjWrapper(model).eval().float()
    for p in wrapper.parameters():
        p.requires_grad_(False)
    dummy = torch.rand(1, 3, VISION_IMAGE_RES, VISION_IMAGE_RES)
    with torch.no_grad():
        shape = wrapper(dummy).shape
    print(f"  probe output shape: {tuple(shape)}")
    path = out / "vision_encoder.onnx"
    torch.onnx.export(
        wrapper, (dummy,), str(path),
        input_names=["pixel_values"], output_names=["image_embeds"],
        dynamic_axes=None, opset_version=OPSET, do_constant_folding=True,
        dynamo=False,
    )
    print(f"  saved {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def export_embed(model, out: Path):
    print("\n--- embed_tokens (ONNX) ---")
    wrapper = EmbedTokens(model).eval().float()
    for p in wrapper.parameters():
        p.requires_grad_(False)
    dummy = torch.randint(0, 100, (1, 8), dtype=torch.long)
    path = out / "embed_tokens.onnx"
    torch.onnx.export(
        wrapper, (dummy,), str(path),
        input_names=["input_ids"], output_names=["text_embeds"],
        dynamic_axes={"input_ids": {1: "n"}, "text_embeds": {1: "n"}},
        opset_version=OPSET, do_constant_folding=True, dynamo=False,
    )
    print(f"  saved {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def export_decoder(decoder, cfg, out: Path):
    print("\n--- Qwen2 decoder w/ KV cache (ONNX) ---")
    decoder = decoder.eval().float()
    for p in decoder.parameters():
        p.requires_grad_(False)
    n_layers = cfg["num_hidden_layers"]
    n_kv = cfg["num_key_value_heads"]
    hd = cfg["hidden_size"] // cfg["num_attention_heads"]
    h = cfg["hidden_size"]

    # trace with a non-trivial past so both prefill (past=0) and decode work
    L, past_len = 3, 2
    embeds = torch.randn(1, L, h)
    pos = torch.arange(past_len, past_len + L)[None]
    past = []
    for _ in range(n_layers):
        past.append(torch.randn(1, n_kv, past_len, hd))
        past.append(torch.randn(1, n_kv, past_len, hd))

    past_names, present_names, dyn = [], [], {
        "inputs_embeds": {1: "seq"},
        "position_ids": {1: "seq"},
        "logits": {1: "seq"},
    }
    for i in range(n_layers):
        for kv in ("key", "value"):
            pn, cn = f"past_{kv}_{i}", f"present_{kv}_{i}"
            past_names.append(pn); present_names.append(cn)
            dyn[pn] = {2: "past_len"}
            dyn[cn] = {2: "total_len"}

    path = out / "decoder.onnx"
    torch.onnx.export(
        decoder, (embeds, pos, *past), str(path),
        input_names=["inputs_embeds", "position_ids", *past_names],
        output_names=["logits", *present_names],
        dynamic_axes=dyn, opset_version=OPSET, do_constant_folding=True,
        dynamo=False,
    )
    print(f"  saved {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def quantize_gather(path: Path):
    """Dynamic INT8 for the embedding table (a Gather -- no activations to
    quantize, so this is effectively weight-only and safe for embeddings)."""
    from onnxruntime.quantization import quantize_dynamic, QuantType
    from onnxruntime.quantization.shape_inference import quant_pre_process
    tmp = path.with_suffix(".pre.onnx")
    try:
        quant_pre_process(str(path), str(tmp), skip_symbolic_shape=True)
        src = tmp
    except Exception as e:
        print(f"  (pre-process skipped: {e})")
        src = path
    quantize_dynamic(
        str(src), str(path),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["Gather"],
        per_channel=True,
    )
    if tmp.exists():
        tmp.unlink()
    print(f"  quantized -> {path.stat().st_size / 1e6:.1f} MB")


def quantize_weight_only(path: Path, bits: int = 8, block_size: int = 128):
    """WEIGHT-ONLY INT8 for the decoder MatMuls (activations stay FP32).

    quantize_dynamic() would ALSO quantize activations, and LLM activation
    outliers blow up per-tensor int8 (we measured ~9.6 max logit error, enough to
    derail greedy decoding on long prompts). Weight-only block-wise quantization
    keeps activations in FP32 and quantizes only the weights -> negligible logit
    error, same ~4x size win.
    """
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        MatMulNBitsQuantizer, DefaultWeightOnlyQuantConfig,
    )
    model = onnx.load(str(path))
    cfg = DefaultWeightOnlyQuantConfig(block_size=block_size, is_symmetric=True, bits=bits)
    quant = MatMulNBitsQuantizer(model, algo_config=cfg)
    quant.process()
    quant.model.save_model_to_file(str(path), use_external_data_format=False)
    print(f"  quantized (weight-only {bits}-bit) -> {path.stat().st_size / 1e6:.1f} MB")


# --------------------------------------------------------------------------- #
#  Verification: standalone decoder vs HF forward                            #
# --------------------------------------------------------------------------- #

def verify_decoder(model, decoder, cfg):
    print("\n--- Verifying standalone decoder against HF forward ---")
    torch.manual_seed(0)
    L = 7
    embeds = torch.randn(1, L, cfg["hidden_size"])
    pos = torch.arange(L)[None]

    model_f = model.float().eval()
    with torch.no_grad():
        hf_out = model_f.model(
            inputs_embeds=embeds, position_ids=pos, use_cache=False, return_dict=True,
        ).last_hidden_state
        hf_logits = model_f.lm_head(hf_out)

    n_kv = cfg["num_key_value_heads"]
    hd = cfg["hidden_size"] // cfg["num_attention_heads"]
    empty = [torch.zeros(1, n_kv, 0, hd) for _ in range(2 * cfg["num_hidden_layers"])]
    decoder = decoder.float().eval()
    with torch.no_grad():
        my_logits = decoder(embeds, pos, *empty)[0]

    diff = (my_logits - hf_logits).abs().max().item()
    rel = diff / (hf_logits.abs().max().item() + 1e-9)
    print(f"  max abs logit diff: {diff:.3e}  (rel {rel:.3e})")
    # also check next-token argmax agreement
    agree = (my_logits.argmax(-1) == hf_logits.argmax(-1)).float().mean().item()
    print(f"  argmax agreement: {agree*100:.1f}%")
    ok = diff < 1e-2 and agree > 0.999
    print("  VERIFY:", "PASS" if ok else "FAIL")
    return ok


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path", nargs="?", default=DEFAULT_MODEL)
    ap.add_argument("--output-dir", default="onnx_models")
    ap.add_argument("--no-quantize", action="store_true", help="FP32 everything")
    ap.add_argument("--quantize-vision", action="store_true", help="INT8 vision too")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    local = Path(args.model_path).expanduser()
    is_local = local.exists()
    model_path = str(local) if is_local else args.model_path
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Model:  {model_path}" + (" (local)" if is_local else " (HF Hub)"))
    print(f"Output: {out}")

    # The training repo (Mobile-O/mobileo) exposes `mobileoForInferenceLM`; the
    # app's self-contained copy exposes `MobileOForInferenceLM`. Accept either.
    try:
        from mobileo.model import mobileoForInferenceLM as InferenceLM
    except ImportError:
        from mobileo.model import MobileOForInferenceLM as InferenceLM
    print("\nLoading checkpoint (fp32)...")
    kw = {"dtype": torch.float32, "low_cpu_mem_usage": True}
    if is_local:
        kw["local_files_only"] = True
    model = InferenceLM.from_pretrained(model_path, **kw).eval()

    c = model.config
    cfg = {
        "hidden_size": c.hidden_size,
        "num_hidden_layers": c.num_hidden_layers,
        "intermediate_size": c.intermediate_size,
        "num_attention_heads": c.num_attention_heads,
        "num_key_value_heads": c.num_key_value_heads,
        "rms_norm_eps": c.rms_norm_eps,
        "vocab_size": c.vocab_size,
        "rope_theta": float(getattr(c, "rope_parameters", {}).get("rope_theta", 1000000.0))
                      if hasattr(c, "rope_parameters") else float(getattr(c, "rope_theta", 1000000.0)),
    }
    print("  cfg:", cfg)

    # build + load standalone decoder from the checkpoint state dict
    sd = model.state_dict()
    decoder = Qwen2Decoder(cfg)
    load_decoder_weights(decoder, sd)

    if not args.skip_verify:
        ok = verify_decoder(model, decoder, cfg)
        if not ok:
            print("\nABORT: standalone decoder does not match HF forward.")
            sys.exit(1)

    # export the three graphs
    vpath = export_vision(model, out)
    epath = export_embed(model, out)
    dpath = export_decoder(decoder, cfg, out)

    # quantize (default: LLM int8, vision fp32)
    if not args.no_quantize:
        print("\n--- Quantizing (INT8 weight-only) ---")
        print("embed_tokens:"); quantize_gather(epath)
        print("decoder:");      quantize_weight_only(dpath)
        if args.quantize_vision:
            print("vision:");   quantize_weight_only(vpath)

    # runtime config + tokenizer
    runtime_cfg = {
        **cfg,
        "num_image_tokens": NUM_IMAGE_TOKENS,
        "image_res": VISION_IMAGE_RES,
        "image_token_sentinel": -200,
        "eos_token_id": int(getattr(c, "eos_token_id", 151645)),
        "bos_token_id": int(getattr(c, "bos_token_id", 151643)),
        "quantized": not args.no_quantize,
        "quantized_vision": bool(args.quantize_vision),
    }
    (out / "runtime_config.json").write_text(json.dumps(runtime_cfg, indent=2))

    src = local if is_local else None
    if src is None:
        from huggingface_hub import snapshot_download
        src = Path(snapshot_download(model_path, allow_patterns=["*token*", "*.txt", "*.jinja", "*.json"]))
    copied = 0
    for fn in TOKENIZER_FILES:
        if (src / fn).exists():
            shutil.copy2(src / fn, out / fn); copied += 1
    print(f"\nCopied {copied} tokenizer/config files.")

    print("\n" + "=" * 60)
    print("ONNX export complete ->", out)
    for p in [vpath, epath, dpath]:
        print(f"  {p.name:22s} {p.stat().st_size / 1e6:8.1f} MB")
    print("=" * 60)
    print("Test it:  python onnx_infer.py --onnx-dir", out, "--image <img> --question '...'")


if __name__ == "__main__":
    main()
