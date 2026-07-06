# Project: Mobile-O VLM Finetuning for GI Hallucination Detection

## Goal
Finetune the VLM (understanding) half of Mobile-O on gastrointestinal endoscopy
data, ultimately using the Gut-VLM hallucination-aware dataset. Kvasir-VQA is
used as a smoke-test dataset to confirm the pipeline works before Gut-VLM arrives.

---

## Current Status
- [x] Smoke test COMPLETE — pipeline confirmed working on Lightning AI
- [x] Loss decreased: 11.25 → 2.69 over 12 steps (200 samples, 1 epoch)
- [x] wandb logging confirmed working on a real Lightning AI run (2026-06-25)
- [~] Full Kvasir-VQA run (58k samples) — now running on Clariden (a168 account,
      /iopsstor/scratch/cscs/dbartaula/FT/). History:
      · Original Lightning AI run: epoch 1 complete (val_loss=0.0495, wandb
        "vocal-flower-2"), epoch 2 died mid-run. epoch_1/ checkpoint recovered
        and pushed to HF Hub (ankitbelbase034/experiments_checkpoints, path
        epoch_1/epoch_1/ — double-nested, use --include "epoch_1/epoch_1/*").
      · Jeevan's Lightning AI account (continuation): started from epoch_1/,
        `--epochs 2 --output_dir checkpoints/vlm_kvasir_full_continued
        --save_every_steps 1000`. Epoch 1 of continuation completed
        (val_loss=0.0495, wandb "vocal-flower-2"), epoch 2 ran to ~14000/14419
        batches then account died. Latest checkpoint uploaded to Clariden was
        from mid-epoch-1 (batch 12000) — NOT from after epoch 1 completed.
      · Clariden run (COMPLETE, 2026-06-26): all 3 epochs finished.
        Results: epoch1 val=0.0442 → epoch2 val=0.0410 → epoch3 val=0.0416
        (epoch 3 overfit slightly). Best checkpoint:
        `checkpoints/vlm_kvasir_full_continued/epoch_2/` (val_loss=0.0410).
        NOTE: `best/` dir was last saved at step 10000 (val_loss≈0.0413,
        worse than epoch_2/) because epoch-end evals didn't compete for
        best/ until fixed in step2_finetune_refined.py 2026-06-26. Always
        prefer epoch_2/ over best/ for this run. READY for step 3.
      · IMPORTANT fix applied 2026-06-26: resume skip was using itertools.islice
        which physically loaded/discarded 12000 batches (blew the 1:30 debug
        time limit). Fixed in step2_finetune_refined.py: make_epoch_loader now
        uses torch.randperm index-slice (O(1) skip, instant). Upload updated
        step2_finetune_refined.py before next run.
- [x] Checkpoint-loading UNEXPECTED-keys scare investigated and resolved — see
      "Checkpoint Key Duplication" section below; `Mobile-O-0.5B-SFT/model.safetensors`
      loads correctly, no retraining needed for that reason
- [x] Gut-VLM dataset — already cloned locally at `Hallucination-Aware-VLM/`
      (top-level, sibling of `Mobile-O/`), annotations present, NOT a future thing anymore
- [x] Gut-VLM data prep script — `Mobile-O/step3_gut_vlm_data.py` (written, verified)
- [x] Hallucination-aware finetuning script — `Mobile-O/step3_finetune_hallucination.py`
      (updated 2026-06-27: added nan_to_num_ on gradients before clip_grad_norm_,
      frozen vision tower set to eval mode, evaluate() returns nan not 0.0 on failure.
      SLURM script: run_step3.sh at top level.)
- [x] Kvasir-v2 images downloaded to Clariden:
      `kvasir-v2-flat/` (flattened from category subdirs in kvasir-dataset-v2.zip)
      URL: https://datasets.simula.no/downloads/kvasir/kvasir-dataset-v2.zip
      (needed --no-check-certificate for SSL)
- [x] Step3 data prep COMPLETE — 1450 train + 366 test JSONL written to
      `Mobile-O/data/gut_vlm/` (0 skipped; images matched by UUID filename)
- [x] Step3 hallucination-aware finetuning COMPLETE (2026-06-27, 6-epoch run)
      Val loss per epoch: 0.1615→0.1454→0.1405→0.1392→0.1392→0.1391
      Converged at epoch 4 (val_loss=0.1392); epochs 5-6 bought <0.0001 improvement.
      Best checkpoint: `checkpoints/vlm_gutvlm_hal/epoch_4/` (val_loss=0.1392)
        or `best/` (val_loss=0.1391, updated at step 480 — negligible difference)
      633M trainable params, ~3 min/epoch on GH200, no overfitting observed.
      NaN fix applied: bfloat16 gradient overflow (Inf→NaN via Inf*clip_coef=0) on
      step 7 without fix; fixed by nan_to_num_(grad) + vision tower .eval() when frozen.
- [x] Inference + Gradio demo scripts written — `Mobile-O/inference.py` + `Mobile-O/app.py`
      (VQA tab + Hallucination Detection tab; run with `python app.py --model_path
      checkpoints/vlm_gutvlm_hal/epoch_4 --share` on Clariden)
- [ ] Run step3 evaluation (QAAS / R-Sim metrics) — see Task 4 below

### IMPORTANT: canonical step2 script moved
`Mobile-O/step2_finetune_vlm.py` is the **original buggy attempt** — kept only
for history, do not use it. The verified, correct version is
`step2_finetune_refined.py` at the **top level** (sibling of `Mobile-O/`, NOT
inside it). It subclasses `mobileoForInferenceLM` and reuses the repo's own
`prepare_inputs_labels_for_multimodal()` instead of hand-injecting image
features — see that file's module docstring for the full reasoning chain
(verified against actual repo source, point by point). Run it with
`MOBILEO_PATH` set to the `Mobile-O/` repo path, or from inside `Mobile-O/`
with the script copied/symlinked in — it does `sys.path.insert` either way.

---

## Repo Structure
```
Finetune_mobile_o/                         ← top level, this is where .claude/ lives
├── .claude/CLAUDE.md                      ← this file
├── step2_finetune_refined.py              ← CANONICAL, verified step2 script (use this one)
├── Hallucination-Aware-VLM/               ← cloned Gut-VLM repo (paper + dataset)
│   ├── dataset/Gut-VLM/
│   │   ├── all_annotations.json           ← all 1816 images, full source format
│   │   ├── VQA_format_all_annotations.json
│   │   ├── Images/                        ← placeholder only (README.md), actual
│   │   │                                     images NOT bundled — download kvasir-v2.zip
│   │   └── train_test_split/
│   │       ├── train.json                 ← 1450 images, SAME format as all_annotations
│   │       ├── test.json                  ← 366 images
│   │       ├── VQA_fromat_trainset_only.json
│   │       └── VQA_format_testset_only.json
│   ├── training_style/hallucinated_aware_train.json  ← paper's own ms-swift-format
│   │                                         example (used to verify our step3_gut_vlm_data.py
│   │                                         output matches exactly — it does)
│   └── eval_scripts/                      ← caption2vqa.py, evaluate_vqa.py, evaluate_caption.py
│                                             (QAAS / R-Sim metrics — not yet ported to our repo)
└── Mobile-O/                               ← must run ALL python scripts from here
    ├── step1_prepare_data.py              ← download Kvasir-VQA, build JSONL
    ├── step2_finetune_vlm.py              ← BUGGY original attempt, do not use
    ├── step3_gut_vlm_data.py              ← converts Gut-VLM annotations → Mobile-O JSONL
    ├── step3_finetune_hallucination.py    ← hallucination-aware finetuning (continues from step2 ckpt)
    ├── inference.py                       ← inference module (load_model, ask, detect_hallucinations)
    ├── app.py                             ← Gradio web demo (VQA + hallucination detection tabs)
    ├── mobileo/                           ← Mobile-O source code (custom, not HF)
    │   ├── model/
    │   │   ├── __init__.py                ← exports: mobileoFastSFTForCausalLM
    │   │   ├── language_model/
    │   │   │   ├── mobileo_sft.py         ← SFT model class (used for training)
    │   │   │   ├── mobileo.py             ← post-train model class (both losses)
    │   │   │   └── mobileo_inference.py   ← inference model (mobile_o_inference type)
    │   │   ├── llava_arch.py              ← LLaVA-style vision integration
    │   │   └── multimodal_llava_encoder/
    │   │       └── mobileclip_encoder.py  ← FastViT vision encoder
    │   ├── train/
    │   │   ├── train.py                   ← SFT training (generation only)
    │   │   ├── post_train.py              ← post-training (gen + und, quadruplet format)
    │   │   └── mobileo_trainer.py         ← custom HF Trainer subclass
    │   ├── mm_utils.py                    ← tokenizer_image_token() utility
    │   └── constants.py                   ← IGNORE_INDEX=-100, IMAGE_TOKEN_INDEX=-200
    ├── checkpoints/
    │   ├── Mobile-O-0.5B-SFT/             ← base checkpoint to finetune FROM
    │   ├── vlm_kvasir/                    ← smoke test output
    │   │   ├── vlm_kvasir/                ← saved model weights
    │   │   └── train_log.jsonl            ← loss log from smoke test
    │   ├── vlm_kvasir_full/                ← Task 1 output
    │   ├── vlm_kvasir_full_continued/     ← continued run; epoch_2/ is step3 base (val=0.0410)
    │   └── vlm_gutvlm_hal/                 ← step3 COMPLETE; epoch_4/ is best (val=0.1392)
    ├── data/
    │   ├── images/                         ← kvasir-VQA images (~6500 jpg files)
    │   ├── smoke_test.jsonl               ← 200 samples used for smoke test
    │   ├── train.jsonl                    ← full 58k Kvasir-VQA samples
    │   └── gut_vlm/                        ← step3_gut_vlm_data.py output (train.jsonl/test.jsonl)
    └── scripts/Mobile-O-0.5B/
        ├── sft.sh                          ← original SFT training script
        └── post_train.sh                   ← original post-training script
```

---

## Critical Architecture Facts

### Model Loading — NEVER use AutoModelForCausalLM
```python
# WRONG — will fail with KeyError: 'mobile_o_inference'
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Amshaker/Mobile-O-0.5B")

# CORRECT — must use Mobile-O's own class
from mobileo.model import mobileoFastSFTForCausalLM
model = mobileoFastSFTForCausalLM.from_pretrained("checkpoints/Mobile-O-0.5B-SFT")
```

### Three Model Classes (different use cases)
| Class | File | Use for |
|---|---|---|
| `mobileoFastSFTForCausalLM` | mobileo_sft.py | Our finetuning (what we use) |
| `mobileoFastForCausalLM` | mobileo.py | Post-training (gen + und jointly) |
| `mobileoForInferenceLM` | mobileo_inference.py | Inference only, loads as `mobile_o_inference` |

### Why We Bypass model.forward()
`mobileoFastSFTForCausalLM.forward()` has `assert latents is not None` — it
ALWAYS runs diffusion loss. We can't use it for understanding-only training.
`mobileoFastForCausalLM` (post-train class) has the same assert.

**CURRENT (correct) solution — `step2_finetune_refined.py` / `step3_finetune_hallucination.py`:**
Subclass `mobileoForInferenceLM` (its `forward()` is never overridden anywhere
in the MRO, so it's literally stock `Qwen2ForCausalLM.forward` — no diffusion
assert at all) and override only `forward()` to call the repo's own
`LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal(..., gen_images=None,
und_images=und_image)`, which splices vision-tower + mm_projector features into
`inputs_embeds` at `IMAGE_TOKEN_INDEX` positions and returns them ready for
`super().forward()`. This reuses the repo's own multimodal wiring instead of
reimplementing it, and correctly handles conversations with more than one
`<image>` tag / multiple turns (the older approach below did not).

**OLD/BUGGY approach — `step2_finetune_vlm.py` (kept for history, do not use):**
1. Call vision tower manually: `model.get_model().get_vision_tower()(und_image)`
2. Call mm_projector: `model.get_model().mm_projector(image_features)`
3. Inject image features into LLM input embeddings at IMAGE_TOKEN_INDEX (-200) positions
   — bug: only replaces the FIRST contiguous image-token span, breaks on
   multi-turn data
4. Call `model.model(inputs_embeds=...)` → Qwen2 LLM only
5. Compute CE loss ourselves with `F.cross_entropy(..., ignore_index=-100)`

### What Gets Frozen vs Trained
```
FROZEN  (594M params): dit, diffusion_connector, sana_vae, noise_scheduler
TRAINED (1070M params): vision_tower (FastViT ~600M) + LLM (Qwen2-0.5B ~500M) + mm_projector
```
To freeze vision tower too (faster, less memory):
Add `"vision_tower"` to `freeze_keywords` in `freeze_generation_components()`.
That gives ~500M trainable instead of 1070M.

### Tokenization
- Mobile-O uses Qwen2 chat format with `<|im_start|>` / `<|im_end|>` tokens
- Image token: `<image>` in text → becomes `IMAGE_TOKEN_INDEX = -200` via `tokenizer_image_token()`
- Labels: human turns masked to `IGNORE_INDEX = -100`, only assistant tokens trained on
- `preprocess_qwen_2()` in post_train.py is the reference implementation

### Data Format (our JSONL)
Single-turn (step1/Kvasir-VQA):
```json
{
  "image": "/abs/path/to/image.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nIs there a polyp in this image?"},
    {"from": "gpt",   "value": "Yes, there is a sessile polyp visible."}
  ]
}
```
4-turn hallucination-aware (step3/Gut-VLM) — same `conversations` schema, just
more turns. The human/gpt-alternating per-message tokenize+mask loop in the
dataset class is turn-count-agnostic, so no code changes were needed to
support this beyond pointing at the right JSONL:
```json
{
  "image": "/abs/path/to/image.jpg",
  "conversations": [
    {"from": "human", "value": "<system framing>\n\n<image>\nCaption: <original_text>\n\nCan you detect which sentences are hallucinated in the given caption?"},
    {"from": "gpt",   "value": "<sentence1> <non-hallucinated>\n<sentence2> <hallucinated>\n..."},
    {"from": "human", "value": "Can you please correct any hallucinated sentences and generate a modified response?"},
    {"from": "gpt",   "value": "Modified caption: <corrections>"}
  ]
}
```
Generated by `Mobile-O/step3_gut_vlm_data.py` from
`Hallucination-Aware-VLM/dataset/Gut-VLM/train_test_split/{train,test}.json`.
Sentence order comes from sorting the `annotations` dict by `start` offset —
do NOT rely on dict insertion order, it's not guaranteed in the source JSON.

### Kvasir-VQA Dataset
- Source: `SimulaMet-HOST/Kvasir-VQA` on HuggingFace
- 6,500 images, ~58,849 QA pairs
- Fields: `img_id`, `image` (PIL), `question`, `answer`, `source`
- Already downloaded to `data/images/` and `data/train.jsonl`

### Gut-VLM Image Matching — plain filename lookup, NOT all Kvasir images annotated
`step3_gut_vlm_data.py`'s `build_record()` does `images_dir / img_id` where
`img_id` is literally the dict key from `train.json`/`test.json` (a uuid.jpg
matching Kvasir-v2's original filenames) — no fuzzy matching, no ID
translation table. If the file isn't found at that path it's silently
skipped (counted in `skipped`, printed at the end) rather than erroring,
which is exactly why the script currently produces empty output (raw
Kvasir-v2 images not yet downloaded — see Task 2 blocker).
Gut-VLM only annotated **1,816 of the ~6,500-8,000 Kvasir images** (1,450
train + 366 test, verified by loading the actual JSON files) — it's a
curated subset, not full coverage. Images outside that 1,816 are simply
irrelevant to step3.

---

## Training Hyperparameters
**step2_finetune_refined.py** (Kvasir-VQA):
```python
BATCH_SIZE   = 4
GRAD_ACCUM   = 4      # effective batch = 16
LR           = 2e-5
WARMUP_RATIO = 0.03
MAX_LEN      = 512
UND_IMAGE_SIZE = 1024
```
**step3_finetune_hallucination.py** (Gut-VLM) — same except:
```python
LR       = 1e-5   # lower — small dataset, overfitting risk
MAX_LEN  = 768     # 4-turn conversations run longer than single QA pairs
```

### Checkpoint Key Duplication — UNEXPECTED-keys load report is HARMLESS (verified 2026-06-25)
On the real Lightning AI run, `from_pretrained()` printed ~80+ `UNEXPECTED`
keys for `checkpoints/Mobile-O-0.5B-SFT`, all prefixed
`base_model.model.model.{vision_tower,dit,diffusion_connector,mm_projector}...`
— that prefix is PEFT's `get_peft_model()` wrapper signature, which looked
like the checkpoint might be an unmerged LoRA save (i.e. vision/diffusion
weights silently NOT loading, training from random init instead).

**Verified harmless** by dumping `model.safetensors` keys directly
(`safetensors.safe_open`): every group has an EXACT matching count of
prefixed vs. unprefixed keys (vision_tower 629/629, dit 548/548,
diffusion_connector 26/26, mm_projector 4/4). So the file contains a
complete, correctly-named copy of every module AND a redundant
`base_model.model.model.`-prefixed duplicate left over from however this
checkpoint was produced. The correctly-named keys load fine; the
UNEXPECTED rows are just the ignored duplicates. The sibling
`mm_projector.bin`/`gen_projector.bin` files in that checkpoint dir are
also redundant (same weights already in `model.safetensors`) — they're
only consumed via a `pretrain_mm_mlp_adapter` + `initialize_vision_modules()`
path that step2/step3 deliberately skip (see docstring point 7 in
step2_finetune_refined.py).
**If this check is ever needed again** (e.g. on a different checkpoint),
the verification script is reusable — see chat history 2026-06-25, or just
rerun the `safe_open` key-count-by-group snippet against any
`model.safetensors`.

## Held-Out Validation / Per-Epoch Checkpoints (added 2026-06-25)
`step2_finetune_refined.py` now carves a fixed-seed (42) `--val_fraction`
slice (default 2%) out of `--data`, **excluded from training entirely**.
After each epoch: runs `evaluate()` (model.eval()/no_grad pass over the val
set), logs `val_loss` to both `train_log.jsonl` (as an `{"epoch_summary":
N, ...}` line) and wandb, and saves a full checkpoint to
`output_dir/epoch_{N}` (in addition to the final `output_dir/vlm_kvasir`
saved after all epochs). Set `--val_fraction 0` to disable and train on
100% of data like before.

**Why:** on the first full Kvasir-VQA attempt, train loss (logged every
optimizer step) plateaued around step ~1000/3678 of epoch 1 (~0.06-0.08,
oscillating) but the script only saved ONE checkpoint at the very end of
all 3 epochs — there was no way to tell whether epochs 2-3 were still
learning or just overfitting/memorizing answer templates before letting
several more GPU-hours run. Train-loss-only is also a bad signal here
because Kvasir-VQA has many short/templated answers (single words), so
per-step CE swings hard between ~0 and ~0.3 depending on answer length —
need the held-out val_loss trend, not single-step train loss, to judge
whether to continue past an epoch.

**How to apply:** after epoch N finishes, check the printed
`Epoch N val_loss (held-out, never trained on): X.XXXX` line (or `analyze_train_log.py`,
or wandb). If val_loss is flat or rising epoch-to-epoch, stop there instead
of running the remaining epochs — they're not buying anything.
`Mobile-O/analyze_train_log.py` (rolling-average train loss viewer,
also prints per-epoch `val_loss` now) is the tool for inspecting
`train_log.jsonl` without needing wandb's UI:
```bash
python analyze_train_log.py checkpoints/vlm_kvasir_full/train_log.jsonl --window 50
```
**Ported to `step3_finetune_hallucination.py` (2026-06-26)** — step3 now
uses the paper's held-out `test.json` (366 images) as its val set via `--val_data`
(instead of carving off a fraction of the 1450-image train set, which would give
only ~29 samples at 2%). eval_every_steps=30 (~3 mid-epoch evals/epoch given
~90 optimizer steps/epoch), save_every_steps=50. Full resume/epoch-N-checkpoint/
best/ tracking all ported.

## Smoke Test Results
- 200 samples, 1 epoch, 12 optimizer steps
- Loss: 11.25 → 9.875 → 6.0 → 5.5 → 6.8 → 3.6 → 4.9 → 2.56 → 6.4 → 4.0 → 6.6 → **2.69**
- LR schedule hit 0.0 at step 12 — expected for 200-sample smoke test, not a bug
- Loss oscillates due to tiny dataset — will smooth with full 58k samples

---

## Compute Setup
- **Current (testing):** Lightning AI studio, A100 or H100
- **Full runs:** Clariden cluster with GH200 (96GB HBM3)
- **No macOS/iOS needed** — CoreML/MLX is only for mobile deployment, not training

## Environment Setup (Lightning AI)
```bash
cd ~/Mobile-O
pip install -e .
pip install timm diffusers datasets pillow tqdm huggingface_hub
# timm is required for MobileCLIP vision encoder
```

---

## Next Immediate Tasks

### Task 1: Full Kvasir-VQA Run (IN PROGRESS — being restarted with val split)
Use the CANONICAL `step2_finetune_refined.py` (top level), NOT
`Mobile-O/step2_finetune_vlm.py`. First attempt was stopped at epoch 1 once
train loss plateaued, to add held-out val_loss tracking before trusting
epochs 2-3 — see "Held-Out Validation" section above. Restart command:
```bash
cd ~/Mobile-O
python ../step2_finetune_refined.py \
    --model_path checkpoints/Mobile-O-0.5B-SFT \
    --data data/train.jsonl \
    --epochs 3 \
    --output_dir checkpoints/vlm_kvasir_full
```
(`--val_fraction` defaults to 0.02, no need to pass it explicitly.)
After each epoch, check the printed `val_loss` (or `epoch_N` checkpoint dirs,
or wandb) before letting the next epoch start — if it's flat/rising, stop
there rather than running the full 3 epochs. This run's final checkpoint
(`checkpoints/vlm_kvasir_full/vlm_kvasir`, or whichever `epoch_N` had the
best val_loss) is what Task 3's hallucination-aware finetuning continues from.

### Task 2: Gut-VLM Data Prep — DONE, but blocked on raw images
`Mobile-O/step3_gut_vlm_data.py` reads
`Hallucination-Aware-VLM/dataset/Gut-VLM/train_test_split/{train,test}.json`
and writes Mobile-O 4-turn JSONL (see Data Format section above). Verified:
its output for image `53a543e6-...jpg` matches the paper's own
`training_style/hallucinated_aware_train.json` example exactly (same tags,
same corrected text).

**Blocker:** the Hallucination-Aware-VLM repo does NOT bundle the actual
Kvasir-v2 images (`dataset/Gut-VLM/Images/` is just a README placeholder).
Must download `kvasir-v2.zip` from https://datasets.simula.no/kvasir/ and
pass its directory to `--images_dir` before this script produces non-empty
output. Filenames in the annotation JSON are uuid.jpg, must match exactly.

```bash
cd Mobile-O
python step3_gut_vlm_data.py \
    --hal_repo ../Hallucination-Aware-VLM \
    --images_dir /path/to/kvasir-v2/images \
    --out_dir data/gut_vlm
```

### Task 3: Hallucination-Aware Finetuning — COMPLETE (2026-06-27)
`Mobile-O/step3_finetune_hallucination.py` (updated 2026-06-27). SLURM script:
`run_step3.sh` at top level. Key design decisions:
- **Full finetune, not LoRA**: domain shift favors full FT over LoRA for small models
- **Starts from `checkpoints/vlm_kvasir_full_continued/epoch_2/`** (val_loss=0.0410)
- **Vision tower frozen** + set to eval mode (frozen BN should use running stats)
- **633M trainable params** (LLM + mm_projector + Mobile Conditioning module)
- **Val set = paper's test.json** (366 images, never trained on)
- Results: epoch1 val≈0.22 → epoch2 val=0.1673 (improving, no overfit)
- Best checkpoint: `checkpoints/vlm_gutvlm_hal/best` (val_loss=0.1673)
- wandb run: "clariden-gut-vlm-2627140"
- Speed: ~5:25 total on GH200, 18.7 samples/sec

**NaN/bfloat16 gradient overflow fix (critical, affects all step3 runs):**
Root cause: during backward pass, some gradient elements overflow to Inf in
bfloat16. `clip_grad_norm_` computes `clip_coef = 1.0/Inf = 0`, then scales
those Inf elements: `Inf * 0 = NaN` (IEEE 754). The NaN propagates through
`optimizer.step()` into the model weights, corrupting all future forward passes.
Fix: call `torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)` on all
gradient tensors BEFORE calling `clip_grad_norm_`. This zeroes overflow elements
so the norm is always finite. Also: frozen vision tower must be explicitly set to
`.eval()` after every `model.train()` call, or BatchNorm uses noisy batch stats
from only 4 images. Both fixes are in step3_finetune_hallucination.py (2026-06-27).

To run eval or a 3rd epoch from best checkpoint:
```bash
cd Mobile-O
# 3rd epoch (if val_loss still improving):
python step3_finetune_hallucination.py \
    --model_path checkpoints/vlm_gutvlm_hal/best \
    --data data/gut_vlm/train.jsonl \
    --val_data data/gut_vlm/test.jsonl \
    --epochs 1
```

### Task 5: Inference + Demo — SCRIPTS READY (2026-06-27)
`Mobile-O/inference.py` + `Mobile-O/app.py` written.

**inference.py** — importable module with three functions:
- `load_model(model_path, device)` — loads `mobileoForInferenceLM` from checkpoint
- `ask(model, tokenizer, ip, image, question)` — single-turn VQA
- `detect_hallucinations(model, tokenizer, ip, image, caption)` — two-turn: returns
  (detection_tags, corrected_caption)

**app.py** — Gradio web demo with two tabs (VQA / Hallucination Detection).

**How to run on Clariden (interactive node or SLURM):**
```bash
cd ~/Mobile-O
pip install gradio          # one-time; already have torch/transformers/mobileo
python app.py --model_path checkpoints/vlm_gutvlm_hal/epoch_4
# → Gradio starts on http://0.0.0.0:7860
```
Then on your local laptop:
```bash
ssh -L 7860:localhost:7860 <clariden-login-node>
# open http://localhost:7860 in browser
```
OR pass `--share` to get a public temporary Gradio URL (no SSH tunnel needed):
```bash
python app.py --model_path checkpoints/vlm_gutvlm_hal/epoch_4 --share
```

**Key inference implementation detail:** `mobileoForInferenceLM.generate()` (line 40
in mobileo_inference.py) does NOT accept `inputs_embeds` — it raises
`NotImplementedError`. The correct call is:
```python
model.generate(input_ids=tokenized_ids, images=und_image, max_new_tokens=N, ...)
```
where `tokenized_ids` is the output of `tokenizer_image_token()` (contains
IMAGE_TOKEN_INDEX = -200 at the `<image>` position). The generate() override
internally calls `prepare_inputs_labels_for_multimodal()` to swap those for image
feature vectors, then calls `super().generate()` with `inputs_embeds`.
Also: it calls `self.to(torch.float32)` before generating — expect 2× memory vs
bfloat16 training (still fine on GH200 96GB).

### Task 4: Evaluation Script — NOT YET PORTED
`Hallucination-Aware-VLM/eval_scripts/` has `caption2vqa.py` → `parse_vqa.py`
→ `evaluate_vqa.py` (QAAS pipeline) and `evaluate_caption.py` (R-Sim, GPT-4o
based). Both rely on calling GPT-4o as a judge — not yet adapted to run
against our Mobile-O checkpoint's outputs. Use the 366-image
`train_test_split/test.json` split for eval, never `train.json`.

## Resume Capability (added 2026-06-25, after the credit-cutoff incident above)
`step2_finetune_refined.py` now survives Lightning AI credit cutoffs without
losing mid-epoch progress:
- `--save_every_steps N` (default 200) saves a full snapshot — model +
  `optimizer.state_dict()` + `scheduler.state_dict()` + exact position
  (`epoch`, `batches_done`, `global_step`) — to `<output_dir>/latest/` every
  N optimizer steps, overwritten in place (plus once at every epoch boundary).
  `trainer_state.pt` inside that dir holds the position/optimizer state;
  `make_epoch_loader()` reseeds the per-epoch shuffle deterministically
  (`seed = 1000 + epoch`) so resume re-derives the exact same batch order and
  uses index-slice (O(1)) to skip already-trained batches.
- `--resume_from <output_dir>/latest` reloads all of the above and continues
  training as if the process never stopped — same `--output_dir` so
  `train_log.jsonl` keeps appending to one continuous history.
- **Also ported to `step3_finetune_hallucination.py` (2026-06-26).**
- Checkpoints made BEFORE this feature existed (e.g. `epoch_1/` from the first
  full-run attempt) have no `trainer_state.pt` — those can only be loaded via
  `--model_path` for a fresh-optimizer continuation, not `--resume_from`.

### Cross-account checkpoint transfer workflow (Lightning AI credits run out)
When a Lightning AI account's credits die mid-run, the fix is NOT to lose the
work — pull the last good checkpoint off that account and push it via the HF
Hub as the transfer medium (much faster than browser download/upload of
multi-GB files):
1. On the dying/dead account (or after re-downloading its checkpoint
   locally): `cd` **directly into the checkpoint folder itself** (e.g.
   `epoch_1/`, or `output_dir/latest/`) before uploading — running the upload
   command from a parent directory will sweep up unrelated project files
   (this happened once: `.claude/`, `Mobile-O/`, repo README assets all got
   uploaded alongside the actual checkpoint).
2. Use `hf upload-large-folder <repo_id> --repo-type model .` (not plain
   `hf upload`) — it's resumable/retry-friendly for multi-GB folders over
   flaky connections; a plain `hf upload` hit `httpx.ReadTimeout` on the
   final commit step on a ~4.7GB payload. Rerunning the same
   `upload-large-folder` command after a failure is safe — it skips
   already-uploaded bytes.
3. On the new account: `hf download <repo_id> --include "<path>/*"
   --local-dir <dest>` — always use `--include` scoped to the exact
   checkpoint path, since shared "scratch" HF repos accumulate unrelated
   clutter from other projects over time (e.g.
   `ankitbelbase034/experiments_checkpoints` already has unrelated
   VITON-HD/diffusion-model checkpoints sitting at its root from past work —
   harmless to ignore, just don't download them by accident).
4. Verify integrity by comparing the downloaded `model.safetensors` byte size
   against the original (not just "did it download without erroring") before
   trusting it as a `--model_path`.
5. A brand-new Lightning AI account starts with NOTHING — re-clone
   `Mobile-O` from `https://github.com/Amshaker/Mobile-O.git`, re-upload the
   standalone top-level scripts (`step2_finetune_refined.py`,
   `analyze_train_log.py`), `pip install -e .` + extra deps, re-download the
   base `Amshaker/Mobile-O-0.5B` checkpoint, re-run `step1_prepare_data.py`
   to rebuild `data/` (faster than re-uploading 6,500 images), THEN pull the
   recovered checkpoint from the HF Hub before resuming/continuing training.

## WandB Logging
Both `step2_finetune_refined.py` and `step3_finetune_hallucination.py` now
log to wandb every optimizer step (in addition to the existing
`train_log.jsonl`): `loss`, `lr`, `grad_norm`, `epoch`, `samples_per_sec`,
`gpu_mem_gb`, plus `epoch_avg_loss` once per epoch. Controlled via
`--wandb_project` / `--wandb_run_name` / `--no_wandb` (disables wandb,
keeps the jsonl log). Default projects: `mobile-o-vlm-finetune` (step2),
`mobile-o-hallucination-finetune` (step3). `wandb` is already in
`Mobile-O/requirements.txt`. Not yet exercised on an actual GPU run — first
real run should confirm `wandb.init()` doesn't need extra config (API key
via `wandb login` or `WANDB_API_KEY` env var) on Lightning AI/Clariden.

---

## Key Papers
1. **Gut-VLM** (hallucination dataset): `bhattarailab/Hallucination-Aware-VLM`
   (cloned locally at `Hallucination-Aware-VLM/`, sibling of `Mobile-O/`)
   - 1,816 Kvasir-v2 images, 1450 train / 366 test (exact split files in
     `dataset/Gut-VLM/train_test_split/{train,test}.json`)
   - Source annotation format per image: `{"original_text": <hallucinated
     VLM report>, "corrections": <expert-corrected report>, "annotations":
     {"<start>-<end>": {"text", "start", "end", "type": "correct"|"incorrect"}}}`
   - Labels: `<hallucinated>` / `<non-hallucinated>` per sentence (maps from
     annotation `type: "incorrect"`/`"correct"` respectively)
   - 2-stage finetuning = ONE multi-turn conversation per image: turn 1 asks
     the model to tag each sentence of `original_text` as hallucinated/not,
     turn 2 asks it to produce the corrected report. Not two separate
     training runs/datasets. Paper used LoRA; we use full FT (see Task 3).
   - Metrics: ROUGE-L, BLEU, METEOR, R-Sim (1-5), QAAS (%)
   - Key result: hallucination-aware FT beats standard FT (90.89% vs 83.07% QAAS)
   - Images NOT bundled in the cloned repo — separately download
     `kvasir-v2.zip` from https://datasets.simula.no/kvasir/

2. **Mobile-O**: `arXiv:2602.20161`
   - FastVLM (FastViT + Qwen2-0.5B) + SANA DiT + MCP projector
   - HF: `Amshaker/Mobile-O-0.5B` (SFT checkpoint we use as base)

---

## Common Errors and Fixes
| Error | Fix |
|---|---|
| `KeyError: 'mobile_o_inference'` | Use `mobileoFastSFTForCausalLM`, not `AutoModelForCausalLM` |
| `ModuleNotFoundError: No module named 'timm'` | `pip install timm` |
| `assert latents is not None` | Don't use `mobileoFastSFTForCausalLM`/`mobileoFastForCausalLM`'s own `forward()` — use the `mobileoForInferenceLM` subclass override in step2_finetune_refined.py / step3_finetune_hallucination.py |
| `initialize_vision_modules` warning | Safe to ignore — vision modules already loaded from checkpoint |
| Loss = NaN | Check `labels[labels != -100]` — may all be masked |
| LR hits 0 early | Expected for tiny smoke test — fine for full run |