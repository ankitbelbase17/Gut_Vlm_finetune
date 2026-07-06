"""
Step 2: Full-finetune the VLM (understanding) half of Mobile-O on Kvasir-VQA.

KEY FACTS from reading the repo:
- Model class to use:  mobileoFastSFTForCausalLM  (from mobileo/train/train.py)
  This is the SFT model - it DOES have the diffusion forward pass baked in
  BUT we can bypass the diffusion loss by passing gen_image=None and
  providing und_image for the understanding forward path.

  ACTUALLY: reading mobileo_sft.py line 200:
    "assert latents is not None" — it ALWAYS expects gen_image.
  So the SFT model forces diffusion. We CANNOT use it for understanding-only.

- The understanding-only path exists in the POST-TRAIN model (mobileoFastForCausalLM)
  BUT that also asserts latents is not None.

- THE CORRECT APPROACH: Use the LLM backbone directly — Qwen2ForCausalLM.
  Mobile-O is just Qwen2 + a vision encoder (FastViT) + a projector (mm_projector).
  We load the checkpoint's LLM/vision weights, build our own forward pass,
  and only compute CE loss on text tokens. No diffusion involved at all.

- This matches what would happen if you loaded the SFT checkpoint and stripped
  the diffusion components — which is exactly what we want.

HOW WE LOAD:
  Mobile-O must be imported from its own repo (trust_remote_code is not enough
  for this custom model type). We add the repo to sys.path and import directly.

PREREQUISITE:
  cd into the Mobile-O repo root before running, OR set MOBILEO_PATH env var.
  The Mobile-O repo must be pip installed: pip install -e .

Run (smoke test, 200 samples, 1 epoch):
  cd /path/to/Mobile-O
  python /path/to/step2_finetune_vlm.py --smoke_test

Run (full):
  cd /path/to/Mobile-O
  python /path/to/step2_finetune_vlm.py --data /path/to/data/train.jsonl --epochs 3
"""

import sys
import os
import json
import copy
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# ── We MUST be inside the Mobile-O repo (or have it installed) ───────────────
MOBILEO_REPO = os.environ.get("MOBILEO_PATH", ".")
if MOBILEO_REPO not in sys.path:
    sys.path.insert(0, MOBILEO_REPO)

# Now we can import Mobile-O's own modules
from mobileo.model import mobileoFastSFTForCausalLM          # SFT model class
from mobileo.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from mobileo.mm_utils import tokenizer_image_token
from mobileo import conversation as conversation_lib
from transformers import AutoProcessor, get_cosine_schedule_with_warmup

# ── Config ───────────────────────────────────────────────────────────────────
DEFAULT_MODEL   = "Amshaker/Mobile-O-0.5B"   # or path to local SFT checkpoint
DEFAULT_DATA    = "data/smoke_test.jsonl"
DEFAULT_OUTDIR  = "checkpoints/vlm_kvasir"
BATCH_SIZE      = 4
GRAD_ACCUM      = 4            # effective batch = 16
LR              = 2e-5
WARMUP_RATIO    = 0.03
MAX_LEN         = 512
UND_IMAGE_SIZE  = 1024         # Mobile-O understanding resolution

# ── Dataset ──────────────────────────────────────────────────────────────────
class KvasirVQADataset(Dataset):
    """
    Reads JSONL from step1.
    Each record: {"image": "/abs/path.jpg", "conversations": [human, gpt]}
    Returns: input_ids, labels, und_image tensor
    """

    def __init__(self, jsonl_path, tokenizer, image_processor):
        self.records         = [json.loads(l) for l in open(jsonl_path)]
        self.tokenizer       = tokenizer
        self.image_processor = image_processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec   = self.records[idx]
        convs = rec["conversations"]     # [{"from":"human",...}, {"from":"gpt",...}]

        # ── Tokenize with label masking (mask human turn) ────────────────────
        # Mobile-O uses the same Qwen2 chat format as post_train.preprocess_qwen_2
        tokens = []
        labels = []
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        eol_id    = 198   # newline token in Qwen2 vocab

        for msg in convs:
            role = "user" if msg["from"] == "human" else "assistant"
            text = f"<|im_start|>{role}\n{msg['value']}"

            # Replace <image> with the special image token index
            msg_ids = tokenizer_image_token(text, self.tokenizer)
            tokens.extend(msg_ids)

            if msg["from"] == "human":
                labels.extend([IGNORE_INDEX] * len(msg_ids))
            else:
                labels.extend(msg_ids)
                tokens.append(im_end_id); tokens.append(eol_id)
                labels.append(im_end_id); labels.append(eol_id)

        # Truncate
        tokens = tokens[:MAX_LEN]
        labels = labels[:MAX_LEN]

        input_ids = torch.tensor(tokens, dtype=torch.long)
        label_ids = torch.tensor(labels, dtype=torch.long)

        # ── Process image ────────────────────────────────────────────────────
        image = Image.open(rec["image"]).convert("RGB")
        und_image = self.image_processor.preprocess(
            image,
            return_tensors="pt",
            size={"height": UND_IMAGE_SIZE, "width": UND_IMAGE_SIZE},
        )["pixel_values"].squeeze(0)    # (C, H, W)

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


# ── Freeze generation components ─────────────────────────────────────────────
def freeze_generation_components(model):
    """
    Freeze DiT, MCP (diffusion_connector), VAE — train VLM only.
    Generation-related param names discovered from the repo code.
    """
    freeze_keywords = [
        "dit",                    # diffusion transformer (DiT)
        "diffusion_connector",    # MCP projector
        "sana_vae",               # VAE
        "noise_scheduler",
    ]
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


# ── Training ─────────────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model using Mobile-O's OWN class (not AutoModel)
    print(f"\nLoading model from: {args.model_path}")
    model = mobileoFastSFTForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    )

    # Initialize vision modules (required by Mobile-O's LLaVA-style arch)
    # This sets up the vision tower (FastViT) and mm_projector
    class FakeModelArgs:
        vision_tower = None                 # loaded from checkpoint config
        mm_vision_select_layer = -2
        vision_tower_pretrained = None
        mm_projector_type = "mlp2x_gelu"
        mm_use_im_start_end = False
        mm_use_im_patch_token = False
        mm_patch_merge_type = "flat"
        mm_vision_select_feature = "patch"
        pretrain_mm_mlp_adapter = None
        pretrain_gen_mlp_adapter = None
        diffusion_name_or_path = None      # skip diffusion init
        vlm_num_layers = 4
        gen_vision_tower = None

    try:
        model.get_model().initialize_vision_modules(model_args=FakeModelArgs(), fsdp=None)
    except Exception as e:
        print(f"[INFO] initialize_vision_modules: {e}")
        print("[INFO] Vision modules may already be initialized from checkpoint — continuing.")

    freeze_generation_components(model)
    model = model.to(device)

    # Load processor / tokenizer
    processor = AutoProcessor.from_pretrained(args.model_path)
    try:
        tokenizer       = processor.tokenizer
        image_processor = processor.image_processor
    except:
        tokenizer       = processor
        image_processor = model.get_model().get_vision_tower().image_processor

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    print(f"\nLoading dataset: {args.data}")
    make_collate = lambda b: collate_fn(b, tokenizer.pad_token_id)
    dataset = KvasirVQADataset(args.data, tokenizer, image_processor)
    loader  = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, collate_fn=make_collate, pin_memory=True,
    )
    print(f"Samples: {len(dataset)} | Batches/epoch: {len(loader)}")

    # Optimizer + scheduler
    optimizer    = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01,
    )
    total_steps  = (len(loader) // GRAD_ACCUM) * args.epochs
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    model.train()
    global_step = 0
    print(f"\nStarting {args.epochs} epoch(s)...")

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for step, batch in enumerate(pbar):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            und_image      = batch["und_image"].to(device, dtype=torch.bfloat16)

            # Forward pass — understanding only.
            # We call the model's LLM directly after injecting image features
            # into the input embeddings. This bypasses the diffusion loss entirely.
            #
            # Mobile-O's prepare_inputs_for_sft expects gen_image too, but we
            # can skip diffusion by calling the underlying Qwen2 model directly
            # after the vision tower processes und_image.

            # Step A: get vision features
            with torch.no_grad() if not any(
                p.requires_grad for p in model.get_model().get_vision_tower().parameters()
            ) else torch.enable_grad():
                image_features = model.get_model().get_vision_tower()(und_image)
                image_features = model.get_model().mm_projector(image_features)

            # Step B: build inputs_embeds with image tokens replaced
            inputs_embeds = model.get_model().embed_tokens(
                input_ids.clamp(min=0)   # replace -200 placeholder with 0 for embedding lookup
            )
            # Replace image token positions (IMAGE_TOKEN_INDEX = -200) with image_features
            image_token_mask = (input_ids == IMAGE_TOKEN_INDEX)
            for b_idx in range(input_ids.shape[0]):
                img_positions = image_token_mask[b_idx].nonzero(as_tuple=True)[0]
                if len(img_positions) == 0:
                    continue
                num_img_tokens = image_features.shape[1]
                # Simple injection: replace first occurrence span
                start = img_positions[0].item()
                end   = min(start + num_img_tokens, inputs_embeds.shape[1])
                feat_len = end - start
                inputs_embeds[b_idx, start:end] = image_features[b_idx, :feat_len]

            # Step C: forward through LLM (Qwen2) — pure CE loss on answer tokens
            from transformers.modeling_outputs import CausalLMOutputWithPast
            output = model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )
            logits = model.lm_head(output.last_hidden_state)

            # CE loss: shift logits/labels by 1
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=IGNORE_INDEX,
            )
            loss = loss / GRAD_ACCUM
            loss.backward()

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                entry = {
                    "step":  global_step,
                    "epoch": epoch + 1,
                    "loss":  round(loss.item() * GRAD_ACCUM, 4),
                    "lr":    scheduler.get_last_lr()[0],
                }
                with open(log_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
                epoch_loss += entry["loss"]
                pbar.set_postfix(loss=f"{entry['loss']:.4f}")

        avg = epoch_loss / max(1, len(loader) // GRAD_ACCUM)
        print(f"Epoch {epoch+1} avg loss: {avg:.4f}")

    # Save
    ckpt = out_dir / "vlm_kvasir"
    model.save_pretrained(str(ckpt))
    processor.save_pretrained(str(ckpt))
    print(f"\nSaved → {ckpt}")
    print(f"Log   → {log_path}")
    print("\n✓ Check train_log.jsonl — loss should decrease step by step.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",  default=DEFAULT_MODEL)
    p.add_argument("--data",        default=DEFAULT_DATA)
    p.add_argument("--output_dir",  default=DEFAULT_OUTDIR)
    p.add_argument("--epochs",      type=int, default=1)
    p.add_argument("--smoke_test",  action="store_true",
                   help="200 samples, 1 epoch — just verify the pipeline works")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.smoke_test:
        args.data   = "data/smoke_test.jsonl"
        args.epochs = 1
    train(args)