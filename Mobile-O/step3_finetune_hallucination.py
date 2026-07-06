"""
Step 3: Hallucination-aware finetuning of Mobile-O's VLM half on Gut-VLM.

Continues directly from the step2 Kvasir-VQA checkpoint (architecture and
training-loop internals are otherwise identical to step2_finetune_refined.py
-- same `mobileoForInferenceLM` subclass override, same
`prepare_inputs_labels_for_multimodal` splicing, same CE-loss-on-assistant-
turns dataset). See that file's module docstring for why this approach
(rather than `mobileoFastSFTForCausalLM`/`mobileoFastForCausalLM`) is correct.

WHY FULL FINETUNE (not LoRA) HERE, EVEN ON A SMALL (1450-EXAMPLE) DATASET:
The Hallucination-Aware-VLM paper uses LoRA, but that's applied to general-
purpose VLMs (LLaVA, Qwen2-VL, etc.) that are already strong on natural
images and only need light adaptation. Mobile-O's vision tower (MobileCLIP/
FastViT) and its 0.5B Qwen2 LLM are comparatively small and were never
trained on endoscopy images at all -- the domain shift from pretraining is
large, which favors continued full finetuning over LoRA's low-rank updates.
Overfitting on the small Gut-VLM set is instead mitigated by:
  - continuing from the step2 Kvasir-VQA checkpoint (58k examples) rather
    than the base SFT checkpoint, so the vision tower + LLM are already
    domain-adapted to endoscopy images before this stage even starts
  - freezing the vision tower by default (`--freeze_vision_tower`, on by
    default here) -- it was already adapted in step2 on the same image
    domain (Kvasir-v2/Kvasir-VQA), so there's little left for it to learn
    from only 1450 more images, and freezing it removes ~600M params from
    the overfitting surface
  - a lower LR and fewer epochs than step2 (see defaults below)
  - evaluating on the paper's held-out test.json (366 images) after every
    epoch (and every --eval_every_steps optimizer steps mid-epoch) to detect
    overfitting before it runs too far

WHAT'S NEW VS. KVASIR-VQA (single-turn QA):
Gut-VLM data (see step3_gut_vlm_data.py) is a 4-message conversation per
image: [human: detect-hallucination prompt + report, gpt: per-sentence
<hallucinated>/<non-hallucinated> tags, human: correction request,
gpt: corrected report]. The dataset class's per-message tokenize/mask loop
already handles an arbitrary number of turns (it just alternates
human/gpt and masks human turns) -- no changes needed there versus step2.

Val set: --val_data (default data/gut_vlm/test.jsonl) uses the paper's own
canonical 366-image test split, not a carve-off from train. At 1450 train
samples a 2% carve-off would be only ~29 examples -- far too small to give
a meaningful val_loss signal.

Run:
    cd ~/Mobile-O
    python step3_finetune_hallucination.py \
        --model_path checkpoints/vlm_kvasir_full_continued/epoch_2 \
        --data data/gut_vlm/train.jsonl \
        --val_data data/gut_vlm/test.jsonl \
        --epochs 2
"""

import sys
import os
import json
import argparse
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from transformers.modeling_outputs import CausalLMOutputWithPast

MOBILEO_REPO = os.environ.get("MOBILEO_PATH", ".")
if MOBILEO_REPO not in sys.path:
    sys.path.insert(0, MOBILEO_REPO)

from mobileo.model import mobileoForInferenceLM
from mobileo.model.language_model.mobileo_inference import mobileoConfig
from mobileo.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from mobileo.mm_utils import tokenizer_image_token
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# ── Config ───────────────────────────────────────────────────────────────────
DEFAULT_MODEL   = "checkpoints/vlm_kvasir_full_continued/epoch_2"
DEFAULT_DATA    = "data/gut_vlm/train.jsonl"
DEFAULT_VAL     = "data/gut_vlm/test.jsonl"
DEFAULT_OUTDIR  = "checkpoints/vlm_gutvlm_hal"
BATCH_SIZE      = 4
GRAD_ACCUM      = 4            # effective batch = 16
LR              = 1e-5         # lower than step2's 2e-5 -- small dataset, overfitting risk
WARMUP_RATIO    = 0.03
MAX_LEN         = 768          # 4-turn conversations run longer than single QA pairs
UND_IMAGE_SIZE  = 1024


# ── Understanding-only training wrapper (identical to step2_finetune_refined.py) ──
class mobileoUnderstandingForTraining(mobileoForInferenceLM):
    config_class = mobileoConfig

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        und_image: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                _target_image_embeds,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                gen_images=None,
                und_images=und_image,
            )

        return super(mobileoForInferenceLM, self).forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )


# ── Dataset (multi-turn; same per-message loop as step2, just more turns) ────
class GutVLMDataset(Dataset):
    """
    Each record: {"image": "/abs/path.jpg", "conversations": [4 messages]}
    (human-detect, gpt-detect, human-correct, gpt-correct) -- see
    step3_gut_vlm_data.py. Loop is turn-count-agnostic: alternating
    human/gpt messages are tokenized and concatenated, human turns masked
    to IGNORE_INDEX, gpt turns kept as labels. Used for both train and val.
    """

    def __init__(self, jsonl_path, tokenizer, image_processor):
        self.records         = [json.loads(l) for l in open(jsonl_path)]
        self.tokenizer       = tokenizer
        self.image_processor = image_processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec   = self.records[idx]
        convs = rec["conversations"]

        tokens = []
        labels = []
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        eol_id    = self.tokenizer.encode("\n", add_special_tokens=False)[0]

        for msg in convs:
            role = "user" if msg["from"] == "human" else "assistant"
            text = f"<|im_start|>{role}\n{msg['value']}"

            msg_ids = tokenizer_image_token(text, self.tokenizer)
            tokens.extend(msg_ids)

            if msg["from"] == "human":
                labels.extend([IGNORE_INDEX] * len(msg_ids))
            else:
                labels.extend(msg_ids)
                tokens.append(im_end_id); tokens.append(eol_id)
                labels.append(im_end_id); labels.append(eol_id)

        tokens = tokens[:MAX_LEN]
        labels = labels[:MAX_LEN]

        input_ids = torch.tensor(tokens, dtype=torch.long)
        label_ids = torch.tensor(labels, dtype=torch.long)

        image = Image.open(rec["image"]).convert("RGB")
        und_image = self.image_processor.preprocess(
            image,
            return_tensors="pt",
            size={"height": UND_IMAGE_SIZE, "width": UND_IMAGE_SIZE},
        )["pixel_values"].squeeze(0)

        return {
            "input_ids": input_ids,
            "labels":    label_ids,
            "und_image": und_image,
        }


def collate_fn(batch, pad_id):
    max_len    = max(b["input_ids"].shape[0] for b in batch)
    input_ids  = torch.stack([
        F.pad(b["input_ids"], (0, max_len - b["input_ids"].shape[0]), value=pad_id)
        for b in batch
    ])
    labels     = torch.stack([
        F.pad(b["labels"], (0, max_len - b["labels"].shape[0]), value=IGNORE_INDEX)
        for b in batch
    ])
    attention_mask = input_ids.ne(pad_id)
    und_images     = torch.stack([b["und_image"] for b in batch])
    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        und_image=und_images,
    )


def evaluate(model, loader, device, freeze_vision_tower=True):
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    with torch.no_grad():
        for batch in loader:
            output = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
                und_image=batch["und_image"].to(device, dtype=torch.bfloat16),
            )
            if not torch.isnan(output.loss) and not torch.isinf(output.loss):
                total_loss += output.loss.item()
                n_batches  += 1
    model.train()
    if freeze_vision_tower:
        model.get_model().get_vision_tower().eval()
    # Return nan (not 0.0) when all batches failed, so callers can distinguish
    # "model is corrupted" from "val_loss is legitimately 0"
    return total_loss / n_batches if n_batches > 0 else float("nan")


def make_epoch_loader(dataset, epoch, collate_fn, skip_batches=0):
    """
    Fresh DataLoader per epoch with a manually-seeded shuffle (seed =
    1000 + epoch) so shuffle order is exactly reproducible on resume.
    skip_batches slices the pre-computed index list -- O(1), no data loading
    during the skip.
    """
    g = torch.Generator()
    g.manual_seed(1000 + epoch)
    indices = torch.randperm(len(dataset), generator=g).tolist()
    if skip_batches:
        indices = indices[skip_batches * BATCH_SIZE:]
    subset = torch.utils.data.Subset(dataset, indices)
    return DataLoader(
        subset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, collate_fn=collate_fn, pin_memory=True,
    )


def save_resume_checkpoint(path, model, tokenizer, optimizer, scheduler,
                            epoch, batches_done, global_step):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))
    tokenizer.save_pretrained(str(path))
    torch.save({
        "optimizer_state":  optimizer.state_dict(),
        "scheduler_state":  scheduler.state_dict(),
        "epoch":            epoch,
        "batches_done":     batches_done,
        "global_step":      global_step,
    }, path / "trainer_state.pt")


def load_resume_state(path):
    state_path = Path(path) / "trainer_state.pt"
    if not state_path.exists():
        return None
    return torch.load(state_path, map_location="cpu")


def freeze_generation_components(model, freeze_vision_tower=True):
    freeze_keywords = [
        "model.dit",
        "model.diffusion_connector",
        "model.vae",
        "model.noise_scheduler",
    ]
    if freeze_vision_tower:
        freeze_keywords.append("model.vision_tower")

    frozen = trainable = 0
    for name, param in model.named_parameters():
        if any(kw in name for kw in freeze_keywords):
            param.requires_grad = False
            frozen += 1
        else:
            param.requires_grad = True
            trainable += 1

    trainable_M = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    total_M     = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Frozen param groups: {frozen} | Trainable param groups: {trainable}")
    print(f"Trainable: {trainable_M:.1f}M / {total_M:.1f}M params")

    # Frozen vision tower must also be in eval mode so BatchNorm uses running
    # statistics instead of noisy batch statistics (batch_size=4 only)
    if freeze_vision_tower:
        model.get_model().get_vision_tower().eval()


def init_wandb(args, run_config):
    if args.no_wandb:
        return None
    import wandb
    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=run_config,
    )


# ── Training ─────────────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    wandb_run = init_wandb(args, {
        "model_path": args.model_path,
        "data": args.data,
        "val_data": args.val_data,
        "epochs": args.epochs,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "lr": LR,
        "warmup_ratio": WARMUP_RATIO,
        "max_len": MAX_LEN,
        "freeze_vision_tower": args.freeze_vision_tower,
        "stage": "gut_vlm_hallucination_aware",
    })

    resume_state = load_resume_state(args.resume_from) if args.resume_from else None
    load_path = args.resume_from if resume_state is not None else args.model_path

    print(f"\nLoading model from: {load_path}")
    model = mobileoUnderstandingForTraining.from_pretrained(
        load_path,
        torch_dtype=torch.bfloat16,
    )
    model = model.to(device)

    freeze_generation_components(model, freeze_vision_tower=args.freeze_vision_tower)

    tokenizer = AutoTokenizer.from_pretrained(load_path)
    image_processor = model.get_model().get_vision_tower().image_processor
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\nLoading dataset: {args.data}")
    make_collate = lambda b: collate_fn(b, tokenizer.pad_token_id)
    dataset = GutVLMDataset(args.data, tokenizer, image_processor)
    steps_per_epoch = len(dataset) // BATCH_SIZE + (1 if len(dataset) % BATCH_SIZE else 0)
    print(f"Train samples: {len(dataset)} | Batches/epoch: {steps_per_epoch}")

    val_loader = None
    if args.val_data and Path(args.val_data).exists():
        val_dataset = GutVLMDataset(args.val_data, tokenizer, image_processor)
        val_loader  = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=2, collate_fn=make_collate, pin_memory=True,
        )
        print(f"Val samples: {len(val_dataset)} (paper's held-out test split, never trained on)")
    else:
        print(f"Val data not found at {args.val_data} -- skipping validation")

    optimizer    = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01,
    )
    total_steps  = max(1, (steps_per_epoch // GRAD_ACCUM) * args.epochs)
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    start_epoch  = 0
    skip_batches = 0
    global_step  = 0
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer_state"])
        scheduler.load_state_dict(resume_state["scheduler_state"])
        start_epoch  = resume_state["epoch"]
        skip_batches = resume_state["batches_done"]
        global_step  = resume_state["global_step"]
        print(f"\nResuming from {args.resume_from}: "
              f"epoch {start_epoch + 1}, batch {skip_batches} of that epoch, "
              f"global_step {global_step}")

    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path    = out_dir / "train_log.jsonl"
    latest_ckpt = out_dir / "latest"
    best_ckpt   = out_dir / "best"
    best_val_loss = float("inf")

    model.train()
    if args.freeze_vision_tower:
        model.get_model().get_vision_tower().eval()
    print(f"\nStarting from epoch {start_epoch + 1} of {args.epochs}...")

    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        epoch_opt_steps = 0
        this_epoch_skip = skip_batches if epoch == start_epoch else 0
        skip_batches = 0

        if this_epoch_skip:
            print(f"Skipping first {this_epoch_skip} already-trained batches of epoch {epoch+1} (index-slice, instant)...")
        loader = make_epoch_loader(dataset, epoch, make_collate, skip_batches=this_epoch_skip)
        batch_iter = enumerate(loader, start=this_epoch_skip)

        optimizer.zero_grad()
        pbar = tqdm(batch_iter, desc=f"Epoch {epoch+1}/{args.epochs}",
                    total=steps_per_epoch, initial=this_epoch_skip)
        step_start = time.time()

        for step, batch in pbar:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            und_image      = batch["und_image"].to(device, dtype=torch.bfloat16)

            # Prevent NaN CE loss when all labels in the batch are masked
            if (labels != IGNORE_INDEX).sum() == 0:
                print(f"[WARNING] batch {step}: all labels masked, skipping")
                continue

            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                und_image=und_image,
            )

            # Guard against NaN/Inf loss (bad sample or numerical instability)
            # Zero accumulated gradients so a bad batch can't corrupt the optimizer state
            if torch.isnan(output.loss) or torch.isinf(output.loss):
                valid = (labels != IGNORE_INDEX).sum().item()
                print(f"[WARNING] NaN/Inf loss at batch {step} "
                      f"(valid_labels={valid}, seq_len={input_ids.shape[1]}), "
                      f"resetting grad accum")
                optimizer.zero_grad()
                continue

            loss = output.loss / GRAD_ACCUM
            loss.backward()

            if (step + 1) % GRAD_ACCUM == 0:
                # Inf gradient elements (bfloat16 overflow) become NaN when
                # clip_grad_norm_ multiplies them by 1/norm=0 (Inf*0=NaN in
                # IEEE 754), which then corrupts model weights via optimizer.step().
                # Zero them out first so clip_grad_norm_ always sees a finite norm.
                for p in model.parameters():
                    if p.requires_grad and p.grad is not None:
                        torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                epoch_opt_steps += 1

                elapsed = time.time() - step_start
                step_start = time.time()
                samples_per_sec = (BATCH_SIZE * GRAD_ACCUM) / max(elapsed, 1e-6)
                gpu_mem_gb = (
                    torch.cuda.memory_allocated(device) / 1e9
                    if torch.cuda.is_available() else 0.0
                )

                entry = {
                    "step":  global_step,
                    "epoch": epoch + 1,
                    "loss":  round(loss.item() * GRAD_ACCUM, 4),
                    "lr":    scheduler.get_last_lr()[0],
                    "grad_norm": float(grad_norm),
                    "samples_per_sec": round(samples_per_sec, 2),
                    "gpu_mem_gb": round(gpu_mem_gb, 2),
                }
                with open(log_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
                if wandb_run is not None:
                    wandb_run.log(entry, step=global_step)
                epoch_loss += entry["loss"]
                pbar.set_postfix(loss=f"{entry['loss']:.4f}")

                if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                    save_resume_checkpoint(
                        latest_ckpt, model, tokenizer, optimizer, scheduler,
                        epoch=epoch, batches_done=step + 1, global_step=global_step,
                    )

                if args.eval_every_steps > 0 and global_step % args.eval_every_steps == 0 and val_loader is not None:
                    step_val_loss = evaluate(model, val_loader, device, args.freeze_vision_tower)
                    is_best = step_val_loss < best_val_loss
                    if is_best:
                        best_val_loss = step_val_loss
                    best_str = " <- best so far" if is_best else ""
                    print(f"[step {global_step}] val_loss={step_val_loss:.4f}{best_str}")
                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "step_eval": global_step,
                            "epoch": epoch + 1,
                            "val_loss": step_val_loss,
                            "is_best": is_best,
                        }) + "\n")
                    if wandb_run is not None:
                        wandb_run.log({"step_val_loss": step_val_loss}, step=global_step)
                    if is_best:
                        model.save_pretrained(str(best_ckpt))
                        tokenizer.save_pretrained(str(best_ckpt))
                        print(f"  Best checkpoint -> {best_ckpt}")

        avg = epoch_loss / max(1, epoch_opt_steps)
        print(f"Epoch {epoch+1} avg loss: {avg:.4f}")

        val_loss = None
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, device, args.freeze_vision_tower)
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            best_str = " <- best so far" if is_best else ""
            print(f"Epoch {epoch+1} val_loss (paper's held-out test split): {val_loss:.4f}{best_str}")
            if is_best:
                model.save_pretrained(str(best_ckpt))
                tokenizer.save_pretrained(str(best_ckpt))
                print(f"  Best checkpoint -> {best_ckpt}")

        with open(log_path, "a") as f:
            f.write(json.dumps({"epoch_summary": epoch + 1, "epoch_avg_loss": avg, "val_loss": val_loss}) + "\n")

        if wandb_run is not None:
            log_entry = {"epoch": epoch + 1, "epoch_avg_loss": avg}
            if val_loss is not None:
                log_entry["val_loss"] = val_loss
            wandb_run.log(log_entry, step=global_step)

        epoch_ckpt = out_dir / f"epoch_{epoch+1}"
        model.save_pretrained(str(epoch_ckpt))
        tokenizer.save_pretrained(str(epoch_ckpt))
        print(f"Checkpoint saved -> {epoch_ckpt}")

        save_resume_checkpoint(
            latest_ckpt, model, tokenizer, optimizer, scheduler,
            epoch=epoch + 1, batches_done=0, global_step=global_step,
        )

    ckpt = out_dir / "vlm_gutvlm_hal"
    model.save_pretrained(str(ckpt))
    tokenizer.save_pretrained(str(ckpt))
    print(f"\nSaved -> {ckpt}")
    print(f"Log   -> {log_path}")
    print("\nCheck val_loss per epoch in train_log.jsonl -- if epoch2 val_loss "
          "rises above epoch1, stop at epoch_1/ rather than using the final checkpoint.")

    if wandb_run is not None:
        wandb_run.finish()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",  default=DEFAULT_MODEL,
                   help="Continue from the step2 Kvasir-VQA best checkpoint.")
    p.add_argument("--data",        default=DEFAULT_DATA)
    p.add_argument("--val_data",    default=DEFAULT_VAL,
                   help="Path to val JSONL. Defaults to the paper's held-out test split "
                        "(data/gut_vlm/test.jsonl, 366 images). Set to empty string to "
                        "disable validation.")
    p.add_argument("--output_dir",  default=DEFAULT_OUTDIR)
    p.add_argument("--epochs",      type=int, default=2,
                   help="Fewer epochs than step2 -- only 1450 examples, higher overfitting risk.")
    p.add_argument("--resume_from", default=None,
                   help="Path to a checkpoint dir saved by this script (e.g. "
                        "<output_dir>/latest) containing trainer_state.pt. "
                        "Resumes model weights + optimizer/scheduler state + "
                        "exact batch position.")
    p.add_argument("--eval_every_steps", type=int, default=30,
                   help="Run validation every N optimizer steps (default 30 -- ~3x per epoch "
                        "given ~90 optimizer steps/epoch at 1450 samples, batch=4, grad_accum=4). "
                        "Set to 0 to disable mid-epoch evaluation.")
    p.add_argument("--save_every_steps", type=int, default=50,
                   help="Save a full resume checkpoint to <output_dir>/latest every N optimizer "
                        "steps. Set to 0 to disable mid-epoch checkpointing.")
    p.add_argument("--freeze_vision_tower", action="store_true", default=True,
                   help="Freeze the vision tower (default: on). It was already domain-adapted "
                        "in step2 on the same image distribution; freezing it here removes "
                        "~600M params from the overfitting surface on this small dataset.")
    p.add_argument("--no_freeze_vision_tower", dest="freeze_vision_tower", action="store_false")
    p.add_argument("--wandb_project", default="mobile-o-hallucination-finetune")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--no_wandb", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
