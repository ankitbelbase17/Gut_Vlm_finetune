#!/usr/bin/env python3
"""
Export the *understanding half* of our finetuned Mobile-O (Gut-VLM) checkpoint
for the iOS app.

Unlike the original Mobile-O app (which exports 5 components), this app only does
image UNDERSTANDING (VQA + hallucination detection/correction), so we export just
the two components the understanding path uses:

  1. Vision Encoder (FastViT)   -> CoreML FP16  (vision_encoder.mlpackage)
  2. LLM (Qwen2-0.5B) + mm_proj  -> MLX 8-bit    (llm/model.safetensors + config.json)

The DiT transformer, VAE decoder and diffusion connector are GENERATION-only and
are deliberately NOT exported - they are never touched by the understanding
forward pass (see FastVLM.swift `prepare()` -> `.logits`).

Default quantization is 8-bit (not 4-bit): our model is a small 0.5B model
finetuned on a narrow medical task, and hallucination detection is
precision-sensitive, so 8-bit is the safer default. Pass `--llm-bits 4` to try
4-bit (smaller download) and compare quality against the Clariden reference.

REQUIREMENTS: must run on macOS (coremltools needs macOS for CoreML conversion).

Usage:
    # Export from our finetuned checkpoint (recommended)
    python export.py /path/to/checkpoints/vlm_gutvlm_hal/epoch_4

    # 4-bit LLM instead of 8-bit
    python export.py /path/to/ckpt --llm-bits 4

    # Only re-export one component
    python export.py /path/to/ckpt --only vision
    python export.py /path/to/ckpt --only llm
"""

import argparse
import json
import logging
import os
import sys
import warnings
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import numpy as np
import torch
import coremltools as ct

# Suppress noisy warnings from dependencies
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
logging.getLogger("coremltools").setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).parent))

# -- Constants ----------------------------------------------------------

VISION_IMAGE_RES = 1024

MM_PROJECTOR_REMAP = {
    "mm_projector.0.": "multi_modal_projector.linear_0.",
    "mm_projector.2.": "multi_modal_projector.linear_2.",
}

# Default to our finetuned hallucination-aware checkpoint. Override with the
# positional `model_path` argument.
DEFAULT_MODEL = "checkpoints/vlm_gutvlm_hal/epoch_4"
ALL_COMPONENTS = ["vision", "llm"]

COMPONENT_OUTPUTS = {
    "vision": ("Vision", "vision_encoder.mlpackage"),
    "llm":    ("LLM",    "llm/model.safetensors"),
}

# -- Helpers -------------------------------------------------------------------

def model_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024**2

def trace_and_freeze(module, example_inputs):
    """Trace a module and freeze the resulting graph."""
    module.eval()
    for p in module.parameters():
        p.requires_grad = False
    with torch.no_grad():
        traced = torch.jit.trace(module, example_inputs, strict=False)
    return torch.jit.freeze(traced)

# -- Export functions ----------------------------------------------------------

def export_vision(model, output_dir: Path):
    """Export the vision encoder to CoreML FP16.

    NOTE: we export the vision tower from OUR checkpoint, not the public base
    model. In step2 (Kvasir-VQA) the vision tower was trained, so its weights
    differ from Amshaker/Mobile-O-0.5B. Reusing the public iOS vision encoder
    would silently pair our finetuned LLM with the wrong image features.
    """
    print("\n--- Vision Encoder (CoreML FP16) ---")

    vision_tower = model.get_model().get_vision_tower()
    if vision_tower is None:
        raise ValueError("Vision tower not found in model")
    if not vision_tower.is_loaded:
        vision_tower.load_model()

    vision_tower.eval().float()
    dummy = torch.rand(1, 3, VISION_IMAGE_RES, VISION_IMAGE_RES)

    traced = trace_and_freeze(vision_tower, (dummy,))

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="images", shape=dummy.shape)],
        outputs=[ct.TensorType(name="image_features", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.ALL,
    )

    path = output_dir / "vision_encoder.mlpackage"
    mlmodel.save(str(path))
    print(f"  Saved: {path} ({model_size_mb(path):.1f} MB)")

def export_llm(model, output_dir: Path, model_path: Path, bits: int = 8):
    """Export LLM + mm_projector to MLX safetensors (default 8-bit)."""
    group_size = 64
    print(f"\n--- LLM (MLX {bits}-bit) ---")

    import mlx.core as mx

    llm_model = model.model
    lm_head = model.lm_head
    vocab_size = lm_head.weight.shape[0]

    llm_config = {
        "model_type": "qwen2",
        "hidden_size": llm_model.config.hidden_size,
        "num_hidden_layers": llm_model.config.num_hidden_layers,
        "intermediate_size": llm_model.config.intermediate_size,
        "num_attention_heads": llm_model.config.num_attention_heads,
        "num_key_value_heads": llm_model.config.num_key_value_heads,
        "rms_norm_eps": llm_model.config.rms_norm_eps,
        "vocab_size": vocab_size,
        "max_position_embeddings": llm_model.config.max_position_embeddings,
        "rope_theta": getattr(llm_model.config, "rope_theta", 1000000),
        "tie_word_embeddings": llm_model.config.tie_word_embeddings,
    }
    print(f"  hidden_size={llm_config['hidden_size']}, layers={llm_config['num_hidden_layers']}, vocab={llm_config['vocab_size']}")

    # Extract state dicts (suppress verbose output)
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
        llm_state = llm_model.state_dict()
        head_state = lm_head.state_dict()

    # Filter to LLM + mm_projector weights
    state_dict = {
        k: v for k, v in llm_state.items()
        if k.startswith(("embed_tokens", "layers.", "norm", "mm_projector"))
    }
    state_dict.update({f"lm_head.{k}": v for k, v in head_state.items()})
    print(f"  Extracted {len(state_dict)} tensors")

    # Remap keys to FastVLM format and convert to MLX
    dtype = mx.float32 if bits == 32 else mx.float16
    weights = {}
    for key, value in state_dict.items():
        arr = mx.array(value.cpu().float().numpy()).astype(dtype)
        if key.startswith("mm_projector."):
            new_key = key
            for old, new in MM_PROJECTOR_REMAP.items():
                new_key = new_key.replace(old, new)
            weights[new_key] = arr
        elif key.startswith("lm_head."):
            weights[f"language_model.{key}"] = arr
        else:
            weights[f"language_model.model.{key}"] = arr

    # Detect mm_hidden_size from projected weights
    mm_proj_key = "multi_modal_projector.linear_0.weight"
    if mm_proj_key not in weights:
        raise ValueError("Could not detect mm_hidden_size from mm_projector weights")
    mm_hidden_size = weights[mm_proj_key].shape[1]

    # Quantize 2D weight tensors, skip small layers (norms, biases)
    if bits in (16, 32):
        quantized_weights = weights
    else:
        print(f"  Quantizing {len(weights)} tensors to {bits}-bit...")
        quantized_weights = {}
        min_elements = 512
        for key, value in weights.items():
            if key.endswith(".weight") and value.ndim == 2 and value.size >= min_elements:
                q_weight, scales, biases = mx.quantize(value, group_size=group_size, bits=bits)
                quantized_weights[key] = q_weight
                quantized_weights[key.replace(".weight", ".scales")] = scales
                quantized_weights[key.replace(".weight", ".biases")] = biases
            else:
                quantized_weights[key] = value

    # Save weights
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_file = output_dir / "model.safetensors"
    mx.save_safetensors(str(weights_file), quantized_weights, metadata={"format": "mlx"})

    total_size = sum(w.nbytes for w in quantized_weights.values())
    index_data = {
        "metadata": {"total_size": int(total_size)},
        "weight_map": {k: weights_file.name for k in quantized_weights},
    }
    with open(output_dir / f"{weights_file.name}.index.json", "w") as f:
        json.dump(index_data, f, indent=4)

    # Save config
    config = {
        **llm_config,
        "model_type": "llava_qwen2",
        "mm_hidden_size": mm_hidden_size,
        "mm_vision_tower": "mobileclip_l_1024",
        "image_token_index": 151648,
    }
    if bits not in (16, 32):
        config["quantization"] = {"group_size": group_size, "bits": bits}
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    # Copy tokenizer files from source model
    import shutil
    tokenizer_files = [
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "added_tokens.json", "merges.txt", "vocab.json",
    ]
    copied = 0
    for fname in tokenizer_files:
        src = model_path / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)
            copied += 1

    print(f"  Saved: {weights_file} ({total_size / 1024**3:.2f} GB)")
    print(f"  Config: {output_dir / 'config.json'}")
    print(f"  Tokenizer: copied {copied} files")
    if copied == 0:
        print("  WARNING: no tokenizer files found next to the checkpoint - the "
              "app needs tokenizer.json etc. Make sure they sit alongside "
              "model.safetensors in the checkpoint dir.")

# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export the understanding half of our finetuned Mobile-O for iOS")
    parser.add_argument("model_path", nargs="?", default=DEFAULT_MODEL, help=f"Local path (or HF id) of the finetuned checkpoint (default: {DEFAULT_MODEL})")
    parser.add_argument("--output-dir", default="exported_models", help="Output directory (default: exported_models)")
    parser.add_argument("--only", nargs="+", choices=ALL_COMPONENTS, help="Export only specific components")
    parser.add_argument("--llm-bits", type=int, default=8, choices=[4, 8, 16, 32], help="LLM quantization bits (default: 8)")
    args = parser.parse_args()

    raw_path = args.model_path
    local_path = Path(raw_path).expanduser()
    is_local = local_path.exists()
    model_path = str(local_path) if is_local else raw_path

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    components = args.only or ALL_COMPONENTS

    if not is_local and "/" not in raw_path:
        print(f"Error: '{raw_path}' is not a local path or a valid HuggingFace model ID")
        sys.exit(1)

    print(f"Model:      {model_path}" + (" (local)" if is_local else " (HuggingFace Hub)"))
    print(f"Output:     {output_dir}")
    print(f"Components: {', '.join(components)}")
    print(f"LLM bits:   {args.llm_bits}")

    from mobileo.model import MobileOForInferenceLM
    print("\nLoading model...")
    load_kwargs = {"dtype": torch.float16}
    if is_local:
        load_kwargs["local_files_only"] = True
    model = MobileOForInferenceLM.from_pretrained(model_path, **load_kwargs)

    if "vision" in components:
        export_vision(model, output_dir)
    if "llm" in components:
        if is_local:
            tokenizer_source = local_path
        else:
            from huggingface_hub import snapshot_download
            tokenizer_source = Path(snapshot_download(model_path, allow_patterns=["tokenizer*", "special_tokens*", "added_tokens*", "merges.txt", "vocab.json"]))
        export_llm(model, output_dir / "llm", model_path=tokenizer_source, bits=args.llm_bits)

    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)
    for c in components:
        label, path = COMPONENT_OUTPUTS[c]
        print(f"  {label + ':':<11}{output_dir}/{path}")
    print("\nNext: upload exported_models/ to your HuggingFace repo (see README).")

if __name__ == "__main__":
    main()
