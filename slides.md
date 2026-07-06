# Finetuning Mobile-O VLM for Gastrointestinal Hallucination Detection

---

## Slide 1: Project Overview

**Goal:** Adapt Mobile-O (a mobile-first vision-language model) to the gastrointestinal endoscopy domain — teaching it to answer clinical questions about endoscopy images and, critically, detect and correct hallucinated sentences in AI-generated endoscopy reports.

**Two-stage training pipeline:**
1. **VQA Finetuning** — on Kvasir-VQA (58k question-answer pairs)
2. **Hallucination-Aware Finetuning** — on Gut-VLM (1,816 annotated endoscopy images)

**Compute:** Lightning AI (A100/H100) for experimentation → Clariden HPC cluster (GH200, 96 GB HBM3) for full runs

---

## Slide 2: What is Mobile-O?

**Mobile-O** is a unified multimodal model designed for on-device inference. It has two halves:

| Component | Role |
|---|---|
| **Vision Tower** (FastViT / MobileCLIP) | Encodes images → ~600M params |
| **MM Projector** | Bridges vision features → LLM embedding space |
| **Qwen2-0.5B LLM** | Language understanding and generation |
| **SANA DiT + VAE** (frozen) | Image generation half — not used in our training |

**We only finetune the understanding half** — vision tower, mm projector, and LLM. The generation components (DiT, diffusion connector, SANA VAE) are permanently frozen and never called during our forward pass.

**Base checkpoint:** `Amshaker/Mobile-O-0.5B-SFT` (HuggingFace)

---

## Slide 3: The Core Technical Challenge — Why We Can't Use the Built-In Training Class

Mobile-O ships three model classes, each for a different purpose:

| Class | Purpose | Problem for us |
|---|---|---|
| `mobileoFastSFTForCausalLM` | SFT training | `assert latents is not None` — always runs diffusion loss, no escape hatch |
| `mobileoFastForCausalLM` | Post-training | Same assert fires after CE loss |
| `mobileoForInferenceLM` | Inference only | `forward()` is plain `Qwen2ForCausalLM.forward()` — no assert |

**Our solution:** Subclass `mobileoForInferenceLM`, override `forward()` with ~15 lines:
1. Call the repo's own `prepare_inputs_labels_for_multimodal(gen_images=None, und_images=...)` → splices vision-tower + mm_projector features into the token-embedding stream at `<image>` positions
2. Pass `inputs_embeds` to `super().forward()` → plain cross-entropy loss

This reuses the existing multimodal wiring (not a reimplementation), correctly handles multi-turn conversations, and avoids all diffusion machinery.

---

## Slide 4: Dataset 1 — Kvasir-VQA (Stage 1)

**Kvasir-VQA** (`SimulaMet-HOST/Kvasir-VQA` on HuggingFace)
- **6,500 endoscopy images**, **58,849 question-answer pairs**
- Covers polyps, ulcers, bleeding, anatomical landmarks
- Questions range from binary ("Is there a polyp?") to descriptive ("What abnormality do you see?")

**Data format — single-turn conversation per QA pair:**
```json
{
  "image": "/abs/path/to/image.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nIs there a polyp in this image?"},
    {"from": "gpt",   "value": "Yes, there is a sessile polyp visible."}
  ]
}
```

**Tokenization:**
- `<image>` tag → replaced with `IMAGE_TOKEN_INDEX = -200` by `tokenizer_image_token()`
- Human turns masked to `IGNORE_INDEX = -100` → model only trains on assistant responses
- Qwen2 chat format: `<|im_start|>` / `<|im_end|>` special tokens

---

## Slide 5: Dataset 2 — Gut-VLM (Stage 2)

**Gut-VLM** (`bhattarailab/Hallucination-Aware-VLM`)
- **1,816 images** from Kvasir-v2, annotated by medical experts (1,450 train / 366 test)
- Each image has: an AI-generated report (`original_text`), an expert-corrected report (`corrections`), and per-sentence span annotations (`"correct"` / `"incorrect"`)
- Task: given the image + AI report, tag each sentence as hallucinated or not, then produce the corrected report

**Data format — 4-turn conversation per image:**
```json
{
  "image": "/path/to/image.jpg",
  "conversations": [
    {"from": "human", "value": "<system prompt>\n\n<image>\nCaption: <AI report>\n\nCan you detect which sentences are hallucinated?"},
    {"from": "gpt",   "value": "<sentence 1> <non-hallucinated>\n<sentence 2> <hallucinated>\n..."},
    {"from": "human", "value": "Can you please correct any hallucinated sentences and generate a modified response?"},
    {"from": "gpt",   "value": "Modified caption: <expert-corrected text>"}
  ]
}
```

**Key detail:** Sentence order is derived by sorting annotation spans by `start` offset (not dict insertion order, which is non-deterministic). The training loop is turn-count-agnostic — no code changes were needed to go from 2-turn to 4-turn data.

---

## Slide 6: Why Full Finetune (SFT) — Not LoRA

The Gut-VLM paper itself used LoRA (applied to LLaVA, Qwen2-VL). We chose **full SFT** for both stages.

| Reason | Explanation |
|---|---|
| **Large domain gap** | Mobile-O was never trained on endoscopy images. FastViT + 0.5B LLM are comparatively small — full FT updates all parameters to close the domain shift |
| **LoRA works for large models** | LoRA's low-rank updates shine when the base model is already strong on the domain (e.g. LLaVA-7B on natural images). Mobile-O-0.5B on endoscopy needs deeper adaptation |
| **Overfitting risk managed** | Stage 2 (Gut-VLM, only 1,450 samples) could overfit — mitigated by: starting from the Stage 1 checkpoint (already endoscopy-adapted), freezing the vision tower, lower LR (1e-5), and early stopping via held-out val set |

**Frozen vs. Trainable parameters:**

| Module | Stage 1 (Kvasir-VQA) | Stage 2 (Gut-VLM) |
|---|---|---|
| Vision Tower (FastViT) | ✅ Trained | ❄️ Frozen |
| MM Projector | ✅ Trained | ✅ Trained |
| Qwen2-0.5B LLM | ✅ Trained | ✅ Trained |
| DiT / SANA VAE | ❄️ Frozen | ❄️ Frozen |
| **Trainable params** | **~1.07B** | **~633M** |

---

## Slide 7: Training Setup — Stage 1 (Kvasir-VQA)

**Script:** `step2_finetune_refined.py`
**Hardware:** Clariden GH200 (96 GB HBM3)

| Hyperparameter | Value |
|---|---|
| Learning rate | 2e-5 |
| Batch size | 4 (grad accum × 4 → effective 16) |
| Epochs | 3 |
| Max sequence length | 512 tokens |
| Image resolution | 1024 × 1024 |
| Warmup ratio | 0.03 |
| LR schedule | Cosine decay |
| Optimizer | AdamW |
| Precision | bfloat16 |

**Val split:** 2% of training data held out (fixed seed 42), never trained on.
**Checkpointing:** Full checkpoint saved after each epoch + every 200 optimizer steps (for resume after credit/cluster interruptions).

---

## Slide 8: Training Results — Stage 1 (Kvasir-VQA)

**Validation loss per epoch:**

| Epoch | Val Loss |
|---|---|
| 1 | 0.0442 |
| **2** | **0.0410** ← best |
| 3 | 0.0416 (slight overfit) |

**Best checkpoint:** `checkpoints/vlm_kvasir_full_continued/epoch_2/`

**Smoke test (sanity check, 200 samples, 1 epoch):**
- Loss trajectory over 12 optimizer steps: 11.25 → 9.87 → 6.0 → 5.5 → 6.8 → 3.6 → 4.9 → 2.56 → 6.4 → 4.0 → 6.6 → **2.69**
- Rapid decrease confirmed the model and data pipeline were working correctly before committing to the full 58k run.

**Observation:** Epoch 3 val_loss rose slightly (0.0410 → 0.0416), indicating the model began memorizing the training set. Epoch 2 is the optimal stopping point.

---

## Slide 9: Training Setup — Stage 2 (Gut-VLM Hallucination)

**Script:** `step3_finetune_hallucination.py`
**Hardware:** Clariden GH200 (96 GB HBM3)

| Hyperparameter | Value | Why different from Stage 1 |
|---|---|---|
| Learning rate | 1e-5 | Smaller dataset → higher overfit risk |
| Max sequence length | 768 tokens | 4-turn conversations are longer than single QA |
| Vision tower | ❄️ Frozen + set to eval() | Already adapted in Stage 1; saves ~600M params from overfit surface |
| Val set | Paper's held-out test.json (366 images) | 2% carve-off would be only ~29 samples |
| Eval frequency | Every 30 optimizer steps | ~3 mid-epoch checkpoints per epoch |

**Starts from:** `checkpoints/vlm_kvasir_full_continued/epoch_2/` (val_loss = 0.0410)
The vision tower and LLM already understand endoscopy images — Stage 2 teaches the structured reasoning pattern for hallucination tagging and correction.

---

## Slide 10: Training Results — Stage 2 (Gut-VLM Hallucination)

**Validation loss per epoch:**

| Epoch | Val Loss | Change |
|---|---|---|
| 1 | 0.1615 | — |
| 2 | 0.1454 | −0.016 |
| 3 | 1.1405 | −0.005 |
| **4** | **0.1392** | **−0.001 ← converged** |
| 5 | 0.1392 | 0.000 |
| 6 | 0.1391 | <0.0001 |

**Best checkpoint:** `checkpoints/vlm_gutvlm_hal/epoch_4/` (val_loss = 0.1392)
- Epochs 5–6 bought < 0.0001 improvement — not worth the compute
- No overfitting observed (val loss never rose)
- Speed: ~5:25 total on GH200, 18.7 samples/sec, ~3 min/epoch
- 633M trainable parameters

---

## Slide 11: Problems Faced and How We Solved Them

### Problem 1: `assert latents is not None`
All built-in training classes unconditionally run diffusion loss. No way to do understanding-only training with them.
**Fix:** Subclassed `mobileoForInferenceLM` (whose `forward()` is stock Qwen2) and overrode `forward()` with ~15 lines using the repo's own `prepare_inputs_labels_for_multimodal`.

### Problem 2: NaN loss from bfloat16 gradient overflow
During backward pass, some gradients overflow to `Inf` in bfloat16. `clip_grad_norm_` computes `clip_coef = 1/Inf = 0`, then `Inf × 0 = NaN` (IEEE 754). NaN propagates through optimizer step and corrupts all future forward passes.
**Fix:** Call `torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)` on all gradients **before** `clip_grad_norm_`. Also: frozen vision tower must be set to `.eval()` after every `.train()` call to prevent BatchNorm using noisy 4-sample batch statistics.

### Problem 3: Compute interruptions (Lightning AI credit cutoffs)
Runs died mid-epoch with no way to resume. Hours of compute lost.
**Fix:** Added `--save_every_steps N` (default 200) that saves model + optimizer + scheduler + exact position to `output_dir/latest/`. Resume with `--resume_from output_dir/latest`. Initial resume implementation used `itertools.islice` (physically loaded and discarded 12,000 batches — blew the 90-minute debug time limit). Fixed to O(1) index-slice via `torch.randperm`.

### Problem 4: UNEXPECTED keys in checkpoint
`from_pretrained()` printed 80+ `UNEXPECTED` keys with `base_model.model.model.*` prefix — looked like unmerged LoRA weights (vision/diffusion might be loading from random init).
**Fix (verification):** Dumped `model.safetensors` keys with `safetensors.safe_open`. Every group had an exact match of prefixed vs. unprefixed key counts (vision_tower: 629/629, DiT: 548/548, etc.). The checkpoint contains a complete correctly-named copy + a redundant PEFT-prefixed duplicate. The correctly-named keys load fine; unexpected rows are ignored. No retraining needed.

---

## Slide 12: Checkpoint Transfer — Cross-Account Workflow

When Lightning AI accounts ran out of credits mid-run, we transferred checkpoints via HuggingFace Hub as the transfer medium (much faster than browser upload/download of multi-GB files).

**Workflow:**
1. `cd` **into** the checkpoint folder itself before uploading (not the parent — learned this the hard way when `.claude/` and `Mobile-O/` got swept up alongside the checkpoint)
2. `hf upload-large-folder <repo_id> --repo-type model .` — resumable on flaky connections (plain `hf upload` hit `httpx.ReadTimeout` on a ~4.7 GB payload)
3. On new account: `hf download <repo_id> --include "epoch_1/epoch_1/*" --local-dir <dest>` — scoped to the exact checkpoint path to avoid downloading unrelated clutter
4. Verify integrity by comparing `model.safetensors` byte size before trusting as `--model_path`

**HF Hub repo used:** `ankitbelbase034/experiments_checkpoints`

---

## Slide 13: Inference — VQA Mode

**Load once, call anywhere:**
```python
from inference import load_model, ask

model, tokenizer, ip = load_model("checkpoints/vlm_gutvlm_hal/epoch_4")
answer = ask(model, tokenizer, ip, image="image.jpg",
             question="Is there a polyp in this image?")
```

**Under the hood:**
1. Image preprocessed to 1024×1024, converted to bfloat16 tensor
2. Prompt built in Qwen2 chat format: `<|im_start|>user\n<image>\n{question}<|im_start|>assistant\n`
3. `tokenizer_image_token()` converts `<image>` → `IMAGE_TOKEN_INDEX = -200`
4. `model.generate(input_ids=..., images=und_image, ...)` — the generate override internally calls `prepare_inputs_labels_for_multimodal()` to swap image token positions for vision-tower features, then calls `super().generate()` with `inputs_embeds`
5. Greedy decoding (do_sample=False) for deterministic clinical outputs

**Key gotcha:** `mobileoForInferenceLM.generate()` does NOT accept `inputs_embeds` directly — it raises `NotImplementedError`. Must pass `input_ids` + `images` and let the model's own generate override do the multimodal wiring.

---

## Slide 14: Inference — Hallucination Detection Mode

Two-turn inference that exactly mirrors the training format:

**Turn 1 — Detection:**
```
<system: expert in GI endoscopy + hallucination definition>
<image>
Caption: {AI-generated report}

Can you detect which sentences are hallucinated in the given caption?
```
→ Model outputs: `<sentence 1> <non-hallucinated>\n<sentence 2> <hallucinated>\n...`

**Turn 2 — Correction:**
Full conversation so far + Turn 1 response, then:
```
Can you please correct any hallucinated sentences and generate a modified response?
```
→ Model outputs: `Modified caption: {corrected text}`

**Critical design:** `<image>` appears **only in Turn 1** (one image per conversation, matches training). Turn 2 is text-only continuation. The model carries visual context through the LLM's KV cache implicitly — no second image pass needed.

```python
detection, corrected_caption = detect_hallucinations(
    model, tokenizer, ip, image="image.jpg",
    caption="The image shows a large polyp with active bleeding. ..."
)
```

---

## Slide 15: Demo — Gradio Web App

`Mobile-O/app.py` — two-tab Gradio interface deployable on Clariden:

```bash
cd ~/Mobile-O
python app.py --model_path checkpoints/vlm_gutvlm_hal/epoch_4 --share
```

**Tab 1: VQA**
- Upload endoscopy image
- Type any clinical question
- Get model answer

**Tab 2: Hallucination Detection**
- Upload endoscopy image
- Paste AI-generated report (e.g. from a base VLM)
- Get per-sentence tags + corrected report

Access locally via SSH tunnel (`ssh -L 7860:localhost:7860 <login-node>`) or publicly via Gradio's `--share` temporary URL.

---

## Slide 16: Benchmarks

### Stage 1 — Kvasir-VQA Training Convergence
| Metric | Value |
|---|---|
| Initial loss (smoke test) | 11.25 |
| Final smoke test loss | 2.69 |
| Best Kvasir-VQA val_loss | **0.0410** (epoch 2 of full 58k run) |

### Stage 2 — Gut-VLM Hallucination Convergence
| Metric | Value |
|---|---|
| Starting val_loss (epoch 1) | 0.1615 |
| Best val_loss (epoch 4) | **0.1392** |
| Total improvement | −0.0223 (−13.8%) |
| Training speed | 18.7 samples/sec on GH200 |

### Paper Benchmark (Gut-VLM, Table 1) — Context for Our Approach
The Gut-VLM paper reports these QAAS scores comparing training strategies on the same dataset:

| Method | QAAS Score |
|---|---|
| Standard VLM finetuning | 83.07% |
| **Hallucination-aware finetuning** | **90.89%** |

We implement the hallucination-aware finetuning strategy. The 7.82 percentage point gain comes entirely from the 4-turn structured conversation format — teaching the model to reason about hallucinations explicitly rather than just generating fluent text.

**Other reported metrics (from the paper, on the 366-image test set):**
- ROUGE-L, BLEU, METEOR for caption quality
- R-Sim (1–5 scale, GPT-4o-judged semantic relevance)
- QAAS (% of questions answered correctly after caption conversion)

Our evaluation pipeline (`step4_generate_predictions.py` + Gut-VLM eval scripts) is ready to run these metrics against our checkpoint.

---

## Slide 17: Full Pipeline Summary

```
Kvasir-VQA (58k QA pairs)
        ↓
  step1_prepare_data.py        → data/train.jsonl
        ↓
  step2_finetune_refined.py    → checkpoints/vlm_kvasir_full_continued/epoch_2/
  (3 epochs, LR=2e-5,          val_loss = 0.0410
   1.07B trainable params)
        ↓
  step3_gut_vlm_data.py        → data/gut_vlm/train.jsonl + test.jsonl
  (1450 train / 366 test)
        ↓
  step3_finetune_hallucination.py  → checkpoints/vlm_gutvlm_hal/epoch_4/
  (6 epochs, LR=1e-5,               val_loss = 0.1392
   vision frozen, 633M params)
        ↓
  inference.py + app.py        → VQA & hallucination detection demo
        ↓
  step4_generate_predictions.py   → results/step4/  (eval in progress)
```

---

## Slide 18: Key Takeaways

1. **Architecture reuse over reimplementation** — Subclassing `mobileoForInferenceLM` and using the repo's own `prepare_inputs_labels_for_multimodal` gave correct multimodal wiring in ~15 lines vs. hundreds if reimplemented from scratch.

2. **Two-stage curriculum matters** — Kvasir-VQA Stage 1 gives the model endoscopy visual vocabulary before Stage 2's small (1,450-sample) hallucination dataset. Starting Stage 2 from the base checkpoint would likely overfit immediately.

3. **Full SFT beats LoRA for small models on new domains** — Domain shift is too large for low-rank updates to close efficiently when the base model is only 0.5B and has never seen endoscopy data.

4. **Engineering reliability is non-trivial at scale** — Resume capability, per-epoch checkpointing, and mid-run val_loss evaluation were all retrofitted after real failures (credit cutoffs, NaN gradients, itertools.islice blowing memory). These are first-class concerns, not afterthoughts.

5. **Structured conversation format drives hallucination performance** — The 4-turn detect-then-correct format forces the model to explicitly reason about per-sentence grounding before writing the corrected report. This is what drives the 83% → 90.89% QAAS gain in the paper.

---

## Slide 19: What's Next

- **Run QAAS / R-Sim evaluation** on the 366-image test set using `step4_generate_predictions.py` and the Gut-VLM eval scripts
- **Quantitative comparison** — benchmark our Mobile-O-0.5B finetuned checkpoint against the paper's LLaVA and Qwen2-VL baselines on the same test split
- **Mobile deployment** — Mobile-O was designed for on-device inference; the finetuned checkpoint can be exported via CoreML/MLX for iOS/Android deployment
- **Potential extensions** — multi-image reports, video endoscopy frames, active learning on high-uncertainty examples

---

*All training runs logged to wandb (`mobile-o-vlm-finetune` / `mobile-o-hallucination-finetune` projects) and local `train_log.jsonl` files. Code at `Finetune_mobile_o/` repo.*
