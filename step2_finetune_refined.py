"""
Step 2: Understanding-only finetuning of Mobile-O's VLM half on Kvasir-VQA.

=============================================================================
WHY THIS SCRIPT LOOKS THE WAY IT DOES (verified against the real repo source)
=============================================================================

1. `mobileoFastSFTForCausalLM` (mobileo/model/language_model/mobileo_sft.py)
   is a dead end for pure understanding SFT:
     - `prepare_inputs_for_sft()` unconditionally calls
       `self.get_model().get_sana_vae()` and encodes `gen_images` through it.
     - `forward()` unconditionally hits
       `assert latents is not None, "Currently we only support image loss
       when latents is None"` right after the LLM forward pass.
   There is no `gen_images=None` escape hatch inside this class. Confirmed
   by reading the class end-to-end — this matches what the project already
   suspected.

2. `mobileoFastForCausalLM` (mobileo/model/language_model/mobileo.py), the
   post-train class, is architecturally what we want to imitate:
       (... , inputs_embeds, labels, latents) = self.prepare_inputs_labels_for_multimodal(
           input_ids, position_ids, attention_mask, past_key_values, labels,
           gen_image, und_image,
       )
       output = super().forward(inputs_embeds=inputs_embeds, labels=labels, ...)
       ce_loss = output.loss
       assert latents is not None   # <-- diffusion branch, still unconditional
   It calls `prepare_inputs_labels_for_multimodal` from `LlavaMetaForCausalLM`
   (mobileo/model/llava_arch.py) and gets a clean `ce_loss` out of
   `super().forward()` BEFORE the diffusion assert fires. We just need to
   stop before that assert.

3. `LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal` (llava_arch.py,
   line ~208) is the actual understanding-image path:
       if (gen_images is None and und_images is None) or ... :
           return input_ids, position_ids, attention_mask, past_key_values, None, labels, None
       if gen_images is not None:
           ... encode through self.get_model().get_sana_vae() ...
       else:
           target_image_embeds = None      # <-- this is our branch
       images = und_images
       image_features = self.visual(images)   # vision_tower -> mm_projector
       ... splice image_features into the token-embedding stream at
           IMAGE_TOKEN_INDEX positions, returns new inputs_embeds/labels ...
   Critically: when `gen_images=None`, the SANA VAE encode call is SKIPPED
   entirely inside this function. The VAE/DiT objects still exist as frozen
   modules on the model (built from the checkpoint's `config.json` at load
   time -- see note below), but they are never invoked during our forward
   pass.

4. `mobileoForInferenceLM` (mobileo/model/language_model/mobileo_inference.py)
   already does exactly the routing we need, in its `.generate()` method:
       (..., inputs_embeds, _, _) = self.prepare_inputs_labels_for_multimodal(
           input_ids, position_ids, attention_mask, None, None,
           und_images=images,
       )
       return super().generate(inputs_embeds=inputs_embeds, ...)
   Its `forward()` is NEVER overridden anywhere in the MRO
   (Qwen2ForCausalLM, LlavaMetaForCausalLM) -- grep confirms the only
   `def forward` in llava_arch.py belongs to the unrelated
   `DiffusionConnector` module. So `mobileoForInferenceLM.forward()` is
   literally stock `Qwen2ForCausalLM.forward()`: plain CE loss over
   `labels`, no diffusion assert, no SANA, no VAE call.

   => Rather than writing a brand-new model class from scratch, this script
      subclasses `mobileoForInferenceLM` and overrides ONLY `forward()` to:
        a) call `prepare_inputs_labels_for_multimodal(..., gen_images=None,
           und_images=und_image)` to get `inputs_embeds`/`labels` with
           image features spliced in,
        b) hand `inputs_embeds` straight to `super().forward()` (stock
           Qwen2 CE loss, ignore_index=-100).
      This is a 15-line override sitting on top of code the repo authors
      already wrote and use elsewhere, instead of a parallel reimplementation
      of vision-tower/projector wiring -- much smaller surface area for bugs.

5. IMPORTANT CAVEAT we verified and cannot eliminate from inside this script:
   `LlavaMetaModel.__init__` (llava_arch.py) builds `self.dit` / `self.vae`
   via `build_sana()` / `build_vae()` (mobileo/model/multimodal_decoder/builder.py)
   whenever the loaded config has `diffusion_name_or_path` set -- and it WILL
   be set, because it was written into config.json by whichever SFT/post-train
   run produced your checkpoint. This happens at `from_pretrained()` time,
   driven by the saved config, regardless of which Python wrapper class you
   load it with (mobileoForInferenceLM included). It costs you:
     - one HF Hub fetch of `Efficient-Large-Model/Sana_600M_512px_diffusers`
       (transformer + vae subfolders) the first time, then it's cached
     - extra VRAM for the frozen DiT + VAE sitting on the model
   It does NOT cost you anything at the loss-computation level: those
   modules are simply never called in the forward pass below, and we
   explicitly zero their `requires_grad` so they receive no gradient and
   are not in the optimizer.

   If you want to avoid the network fetch entirely, the only way is to hand
   -edit the checkpoint's config.json to remove the `diffusion_name_or_path`
   key before loading -- see the printed instructions at the bottom of this
   file under "OPTIONAL: skip SANA/VAE network fetch".

6. CONFIG FIELDS double-checked against the real builders:
   - `multimodal_llava_encoder/mobileclip_encoder.py` line 20:
         self.input_image_size = int(vision_tower.split("_")[-1])
     so `vision_tower` MUST end in a numeric resolution suffix, e.g. `_1024`.
   - `multimodal_llava_encoder/mobileclip/__init__.py` line 19:
         model_name = "_".join(model_name.split("_")[0:2])
     which strips a 3-part name like "mobileclip_l_1024" down to
     "mobileclip_l" to find the config file. The ONLY config file shipped
     in the repo is `mobileclip/configs/mobileclip_l.json`. There is no
     "mobileclip_1" or "mobileclip" config. So the correct vision tower
     string is `"mobileclip_l_1024"`, NOT `"mobileclip_1024"` as in the
     original draft -- `"mobileclip_1024".split("_")[0:2]` -> `["mobileclip",
     "1024"]` -> joined as `"mobileclip_1024"` -> no matching config file ->
     crash. This only matters if you are building vision modules from
     scratch; since we load an already-initialized checkpoint, the tower
     and its config come from the checkpoint itself and this string is
     never actually constructed by us (see point 7).
   - `mobileclip/configs/mobileclip_l.json`: `image_cfg.embed_dim = 3072`,
     confirming `mm_hidden_size=3072` is correct.
   - Qwen2-0.5B `hidden_size=896` is correct (also confirmed by
     `mobile_block.py`'s own docstring example: "input_dim=896, # VLM
     output dimension").

7. We deliberately do NOT call `initialize_vision_modules()` and do NOT
   construct a `FakeModelArgs` namespace. That function is for building
   vision/diffusion modules from scratch on top of a base LLM checkpoint
   that doesn't have them yet (used by `pre_train.sh`'s initial run). Your
   checkpoint (`checkpoints/Mobile-O-0.5B-SFT`) already has a fully
   initialized vision tower, projector, DiT, and VAE baked into its
   `config.json` + `model.safetensors` -- `from_pretrained()` reconstructs
   all of it for you. Calling `initialize_vision_modules()` again on top of
   an already-loaded checkpoint is redundant at best, and at worst silently
   double-initializes / reloads modules (it has internal "already loaded"
   guards, but you saw `[INFO] initialize_vision_modules: ...` warnings
   firing in the original smoke test for exactly this reason). Deleting that
   call removes a class of bugs without losing anything.

Run (smoke test, 200 samples, 1 epoch):
    cd ~/Mobile-O
    python step2_finetune_vlm.py --smoke_test

Run (full):
    cd ~/Mobile-O
    python step2_finetune_vlm.py --data data/train.jsonl --epochs 3
"""

import sys
import os
import json
import argparse
import random
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

# ── We MUST be inside the Mobile-O repo (or have it installed) ───────────────
MOBILEO_REPO = os.environ.get("MOBILEO_PATH", ".")
if MOBILEO_REPO not in sys.path:
    sys.path.insert(0, MOBILEO_REPO)

# `mobileoForInferenceLM` is the class with NO diffusion assert anywhere in
# its forward() -- see point 4 in the module docstring above.
from mobileo.model import mobileoForInferenceLM
from mobileo.model.language_model.mobileo_inference import mobileoConfig
from mobileo.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from mobileo.mm_utils import tokenizer_image_token
from transformers import AutoProcessor, AutoTokenizer, get_cosine_schedule_with_warmup

# ── Config ───────────────────────────────────────────────────────────────────
DEFAULT_MODEL   = "checkpoints/Mobile-O-0.5B-SFT"
DEFAULT_DATA    = "data/smoke_test.jsonl"
DEFAULT_OUTDIR  = "checkpoints/vlm_kvasir"
BATCH_SIZE      = 4
GRAD_ACCUM      = 4            # effective batch = 16
LR              = 2e-5
WARMUP_RATIO    = 0.03
MAX_LEN         = 512
UND_IMAGE_SIZE  = 1024         # Mobile-O understanding resolution


# ── Understanding-only training wrapper ───────────────────────────────────────
class mobileoUnderstandingForTraining(mobileoForInferenceLM):
    """
    Thin training wrapper around `mobileoForInferenceLM`.

    We override ONLY forward(). Everything else (config class, __init__,
    get_model, generate, sample_images, etc.) is inherited unchanged from
    `mobileoForInferenceLM`, which already builds the vision tower +
    mm_projector + (frozen) DiT/VAE from the checkpoint at load time via
    `LlavaMetaModel.__init__` / `from_pretrained()`.

    forward() routes und_image through
    `LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal(...,
    gen_images=None, und_images=und_image)`, which:
      - skips the SANA VAE encode path entirely (gen_images is None)
      - runs the image through `self.visual()` ->
        `vision_tower(...)` -> `mm_projector(...)`
      - splices the resulting image features into the embedding stream at
        IMAGE_TOKEN_INDEX positions
      - returns ready-to-use `inputs_embeds` / `labels`
    `inputs_embeds` then goes straight into `super().forward()`
    (= stock `Qwen2ForCausalLM.forward`), which computes standard
    shifted cross-entropy loss with `ignore_index=-100` over `labels`.
    No diffusion transformer, scheduler, or VAE is touched.
    """

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
                _target_image_embeds,   # always None: gen_images=None
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                gen_images=None,
                und_images=und_image,
            )

        # Stock Qwen2ForCausalLM.forward: shifted CE loss, ignore_index=-100,
        # no diffusion branch anywhere in this call.
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


# ── Dataset ──────────────────────────────────────────────────────────────────
class KvasirVQADataset(Dataset):
    """
    Reads JSONL from step1.
    Each record: {"image": "/abs/path.jpg", "conversations": [human, gpt]}
    Returns: input_ids, labels, und_image tensor
    """

    def __init__(self, records, tokenizer, image_processor):
        self.records         = records
        self.tokenizer       = tokenizer
        self.image_processor = image_processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec   = self.records[idx]
        convs = rec["conversations"]     # [{"from":"human",...}, {"from":"gpt",...}]

        tokens = []
        labels = []
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        eol_id    = self.tokenizer.encode("\n", add_special_tokens=False)[0]

        for msg in convs:
            role = "user" if msg["from"] == "human" else "assistant"
            text = f"<|im_start|>{role}\n{msg['value']}"

            # Replace <image> with IMAGE_TOKEN_INDEX (-200)
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
        )["pixel_values"].squeeze(0)    # (C, H, W)

        return {
            "input_ids": input_ids,
            "labels":    label_ids,
            "und_image": und_image,
        }


def load_records(jsonl_path):
    return [json.loads(l) for l in open(jsonl_path)]


def split_train_val(records, val_fraction, seed=42):
    if val_fraction <= 0:
        return records, []
    shuffled = records[:]
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    return shuffled[n_val:], shuffled[:n_val]


def evaluate(model, loader, device):
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
            total_loss += output.loss.item()
            n_batches  += 1
    model.train()
    return total_loss / max(1, n_batches)


def make_epoch_loader(dataset, epoch, collate_fn, skip_batches=0):
    """
    Fresh DataLoader per epoch with a manually-seeded shuffle (seed =
    1000 + epoch) so shuffle order is exactly reproducible on resume.
    skip_batches slices the pre-computed index list -- O(1), no data loading
    during the skip (contrast with itertools.islice which has to load and
    discard every skipped batch from disk).
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
    """
    Full mid-training snapshot: model weights + optimizer/scheduler state +
    exact position (epoch, batches_done = absolute batch index already
    consumed in that epoch's loader, global_step = optimizer steps so far).
    Overwrites `path` in place every time -- this is the "latest" pointer,
    not a history of checkpoints (epoch_N/ checkpoints still serve that role).
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))
    tokenizer.save_pretrained(str(path))
    torch.save({
        "optimizer_state":  optimizer.state_dict(),
        "scheduler_state":  scheduler.state_dict(),
        "epoch":            epoch,          # 0-indexed, epoch IN PROGRESS
        "batches_done":     batches_done,    # absolute batch index within this epoch's loader
        "global_step":      global_step,
    }, path / "trainer_state.pt")


def load_resume_state(path):
    state_path = Path(path) / "trainer_state.pt"
    if not state_path.exists():
        return None
    return torch.load(state_path, map_location="cpu")


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
def freeze_generation_components(model, freeze_vision_tower=False):
    """
    Freeze DiT, MCP (diffusion_connector), VAE, noise_scheduler.
    These are built automatically inside `LlavaMetaModel.__init__` from the
    checkpoint's config.json (see module docstring point 5) -- we just make
    sure they never receive gradients and never enter the optimizer.
    """
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


# ── Training ─────────────────────────────────────────────────────────────────
def init_wandb(args, run_config):
    if args.no_wandb:
        return None
    import wandb
    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=run_config,
    )


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    wandb_run = init_wandb(args, {
        "model_path": args.model_path,
        "data": args.data,
        "epochs": args.epochs,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "lr": LR,
        "warmup_ratio": WARMUP_RATIO,
        "max_len": MAX_LEN,
        "freeze_vision_tower": args.freeze_vision_tower,
        "val_fraction": args.val_fraction,
    })

    resume_state = load_resume_state(args.resume_from) if args.resume_from else None
    load_path = args.resume_from if resume_state is not None else args.model_path

    print(f"\nLoading model from: {load_path}")
    print("(this will also build the frozen DiT/VAE from the checkpoint's "
          "diffusion_name_or_path the first time -- see module docstring "
          "point 5 if you want to skip that network fetch)")
    model = mobileoUnderstandingForTraining.from_pretrained(
        load_path,
        torch_dtype=torch.bfloat16,
    )
    model = model.to(device)

    freeze_generation_components(model, freeze_vision_tower=args.freeze_vision_tower)

    # Load processor / tokenizer
    tokenizer = AutoTokenizer.from_pretrained(load_path)
    image_processor = model.get_model().get_vision_tower().image_processor
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\nLoading dataset: {args.data}")
    make_collate = lambda b: collate_fn(b, tokenizer.pad_token_id)
    train_records, val_records = split_train_val(load_records(args.data), args.val_fraction)

    dataset = KvasirVQADataset(train_records, tokenizer, image_processor)
    steps_per_epoch = len(dataset) // BATCH_SIZE + (1 if len(dataset) % BATCH_SIZE else 0)
    print(f"Train samples: {len(dataset)} | Batches/epoch: {steps_per_epoch}")

    val_loader = None
    if val_records:
        val_dataset = KvasirVQADataset(val_records, tokenizer, image_processor)
        val_loader  = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=2, collate_fn=make_collate, pin_memory=True,
        )
        print(f"Val samples: {len(val_dataset)} (held out, never trained on)")

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
    log_path = out_dir / "train_log.jsonl"
    latest_ckpt = out_dir / "latest"
    best_ckpt   = out_dir / "best"
    best_val_loss = float("inf")

    model.train()
    print(f"\nStarting from epoch {start_epoch + 1} of {args.epochs}...")

    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        epoch_opt_steps = 0
        this_epoch_skip = skip_batches if epoch == start_epoch else 0
        skip_batches = 0   # only the resumed epoch skips; later epochs start fresh

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

            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                und_image=und_image,
            )
            loss = output.loss / GRAD_ACCUM
            loss.backward()

            if (step + 1) % GRAD_ACCUM == 0:
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
                    step_val_loss = evaluate(model, val_loader, device)
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
            val_loss = evaluate(model, val_loader, device)
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            best_str = " <- best so far" if is_best else ""
            print(f"Epoch {epoch+1} val_loss (held-out, never trained on): {val_loss:.4f}{best_str}")
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

    ckpt = out_dir / "vlm_kvasir"
    model.save_pretrained(str(ckpt))
    tokenizer.save_pretrained(str(ckpt))
    print(f"\nSaved -> {ckpt}")
    print(f"Log   -> {log_path}")
    print("\nCheck train_log.jsonl -- loss should decrease step by step.")
    print("Check epoch_N/val_loss in train_log.jsonl or wandb -- if val_loss stops "
          "decreasing (or rises) across epochs, later epochs are likely overfitting "
          "rather than generalizing further.")

    if wandb_run is not None:
        wandb_run.finish()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",  default=DEFAULT_MODEL)
    p.add_argument("--data",        default=DEFAULT_DATA)
    p.add_argument("--output_dir",  default=DEFAULT_OUTDIR)
    p.add_argument("--val_fraction", type=float, default=0.02,
                   help="Fraction of --data held out (fixed seed, never trained on) for "
                        "per-epoch val_loss. Set to 0 to disable and train on 100%% of data.")
    p.add_argument("--epochs",      type=int, default=1)
    p.add_argument("--resume_from", default=None,
                   help="Path to a checkpoint dir saved by this script (e.g. "
                        "<output_dir>/latest) containing trainer_state.pt. "
                        "Resumes model weights + optimizer/scheduler state + "
                        "exact batch position -- continues training as if the "
                        "process never stopped, mid-epoch included. Logs append "
                        "to the SAME --output_dir's train_log.jsonl, so use the "
                        "same --output_dir you used originally.")
    p.add_argument("--eval_every_steps", type=int, default=500,
                   help="Run validation and save to <output_dir>/best if val_loss improves, "
                        "every N optimizer steps. Set to 0 to disable mid-epoch evaluation.")
    p.add_argument("--save_every_steps", type=int, default=200,
                   help="Save a full resume checkpoint (model+optimizer+scheduler+"
                        "position) to <output_dir>/latest every N optimizer steps. "
                        "Set to 0 to disable mid-epoch checkpointing (epoch-boundary "
                        "checkpoints are still saved).")
    p.add_argument("--freeze_vision_tower", action="store_true",
                   help="Also freeze the MobileCLIP/FastViT vision tower "
                        "(~600M params) -- trains only the LLM + projector "
                        "(~500M params), faster/less memory.")
    p.add_argument("--smoke_test",  action="store_true",
                   help="200 samples, 1 epoch -- just verify the pipeline works")
    p.add_argument("--wandb_project", default="mobile-o-vlm-finetune")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--no_wandb", action="store_true",
                   help="Disable wandb logging; only write train_log.jsonl")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.smoke_test:
        args.data   = "data/smoke_test.jsonl"
        args.epochs = 1
    train(args)


# =============================================================================
# OPTIONAL: skip SANA/VAE network fetch entirely
# =============================================================================
# If you'd rather not download Efficient-Large-Model/Sana_600M_512px_diffusers
# at all (e.g. air-gapped Clariden run), edit the checkpoint's config.json
# once, before training:
#
#   import json
#   cfg_path = "checkpoints/Mobile-O-0.5B-SFT/config.json"
#   cfg = json.load(open(cfg_path))
#   cfg.pop("diffusion_name_or_path", None)
#   json.dump(cfg, open(cfg_path, "w"), indent=2)
#
# With that key removed, `LlavaMetaModel.__init__`'s
# `if hasattr(config, "diffusion_name_or_path"):` guard is False, so dit/vae/
# diffusion_connector/noise_scheduler are never constructed at all -- saving
# load time and VRAM. Keep a backup of the original config.json first, since
# you'll want diffusion_name_or_path back if you ever load this checkpoint
# for image generation again.