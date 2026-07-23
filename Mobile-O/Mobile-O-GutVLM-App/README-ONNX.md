# GutVLM on ONNX — runs on any laptop (Windows / Linux / Intel-Mac / Apple-Mac)

The macOS build (`export.py` → CoreML + MLX, driven by `mlx_app.py`) only runs on
Apple Silicon. This ONNX build is the cross-platform sibling: the same finetuned
Gut-VLM understanding model (VQA + hallucination detection/correction), exported to
**ONNX Runtime** so it runs on the plain CPU of any laptop — no CoreML, no MLX, no
GPU, and no PyTorch at inference time.

## What gets exported

The understanding forward pass is three pieces (the DiT / VAE / diffusion
generation stack is never touched, exactly like the MLX build):

| Graph | Input → Output | Precision | Size |
|-------|----------------|-----------|------|
| `vision_encoder.onnx` | `pixel_values[1,3,1024,1024]` → projected image embeds `[1,256,896]` (vision tower **+ mm_projector folded in**) | FP32 | ~505 MB |
| `embed_tokens.onnx` | `input_ids[1,n]` → text embeds `[1,n,896]` | INT8 | ~136 MB |
| `decoder.onnx` | `inputs_embeds` + `position_ids` + past KV → `logits` + present KV (Qwen2, with KV cache) | INT8 (weight-only) | ~510 MB |

**Total ≈ 1.1 GB.** Plus `tokenizer.json`, `runtime_config.json` etc. get copied
alongside.

### Quantization: FP32 vision + INT8 (weight-only) LLM

- The **vision encoder stays FP32** — hallucination detection is precision-sensitive
  (same argument `export.py` makes for defaulting the MLX LLM to 8-bit not 4-bit).
- The **LLM is INT8 _weight-only_** (`MatMulNBitsQuantizer`, block-wise, symmetric).
  This is the important detail: plain `quantize_dynamic` also quantizes
  *activations*, and Qwen2's activation outliers blow up per-tensor int8 — we
  measured a **max logit error of ~9.6**, enough to derail greedy decoding on long
  prompts (VQA still looked fine, but hallucination detection went degenerate).
  Weight-only int8 keeps activations in FP32 and drops the error to **~0.31**, which
  reproduces the FP32 model **token-for-token** on our test prompts.

## Install

```bash
pip install -r requirements-onnx.txt        # onnxruntime + transformers + pillow
```

## Run inference

```bash
# VQA
python onnx_infer.py --onnx-dir onnx_models \
    --image /path/to/endoscopy.jpg \
    --question "Are there any abnormalities in the image?"

# Hallucination detection + correction (two-turn)
python onnx_infer.py --onnx-dir onnx_models \
    --image /path/to/endoscopy.jpg \
    --mode hallucination \
    --caption "The image shows a large polyp with active bleeding."
```

Or from Python (mirrors `inference.py` / `mlx_infer.py` one-to-one):

```python
from onnx_infer import load_state, ask, detect_hallucinations
state = load_state("onnx_models")
print(ask(state, "img.jpg", "Is there a polyp visible?"))
detection, correction = detect_hallucinations(state, "img.jpg", caption)
```

## Re-export from a checkpoint

Needs PyTorch + the `mobileo` package (the exporter accepts either the training
repo's `mobileoForInferenceLM` or the app's `MobileOForInferenceLM`). Run it from a
directory where `import mobileo` resolves to a package that can load the checkpoint
(e.g. the `Mobile-O/` training repo):

```bash
pip install torch onnx onnxscript onnxruntime coremltools   # coremltools not used here
cd /path/to/Mobile-O                                        # where `mobileo` lives

python Mobile-O-GutVLM-App/onnx_export.py \
    checkpoints/vlm_gutvlm_hal/epoch_4 \
    --output-dir Mobile-O-GutVLM-App/onnx_models
# or point at the HuggingFace checkpoint id instead of the local path
```

Useful flags:
- `--no-quantize` — FP32 everything (~2.7 GB), the correctness baseline.
- `--quantize-vision` — INT8 the vision encoder too (smaller, riskier on the
  precision-sensitive medical task).
- `--skip-verify` — skip the numeric check below.

**Built-in verification.** Before exporting, `onnx_export.py` re-implements the
Qwen2 decoder from scratch (transformers 5.x's KV-cache `Cache` API does not trace
cleanly) and checks it against the real HuggingFace forward pass. On this checkpoint
it agrees to `max abs logit diff ≈ 3e-5`, `argmax agreement 100%` — so the standalone
decoder is numerically the same model, just ONNX-friendly.

## Notes / limitations

- **CPU greedy decoding.** ~0.5B params in INT8; expect a few tokens/sec on a
  typical laptop CPU. Use `--threads N` to pin intra-op threads. For a GPU laptop
  you can swap in `CUDAExecutionProvider`/`DmlExecutionProvider` in `onnx_infer.py`.
- **Fixed 1024×1024 image input** (matches training); the vision graph is static-shape.
- Only greedy (`do_sample=False`) is implemented, matching `inference.py`.
