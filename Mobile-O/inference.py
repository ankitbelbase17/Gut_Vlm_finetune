"""
Inference module for the finetuned Mobile-O hallucination-detection checkpoint.

Supports two modes:
  1. VQA (single-turn): upload an endoscopy image, ask any question.
  2. Hallucination detection (two-turn): give an image + AI-generated caption;
     the model tags each sentence as <hallucinated>/<non-hallucinated> and then
     produces a corrected caption.  This matches the Gut-VLM training format
     exactly.

Usage (command-line):
    cd ~/Mobile-O

    # VQA
    python inference.py \\
        --model_path checkpoints/vlm_gutvlm_hal/epoch_4 \\
        --image /path/to/image.jpg \\
        --question "Is there a polyp visible in this endoscopy image?"

    # Hallucination detection
    python inference.py \\
        --model_path checkpoints/vlm_gutvlm_hal/epoch_4 \\
        --image /path/to/image.jpg \\
        --mode hallucination \\
        --caption "The image shows a large polyp with active bleeding.
There is a visible instrument in the frame."

Import and use in app.py or other scripts:
    from inference import load_model, ask, detect_hallucinations
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Union

import torch
from PIL import Image
from transformers import AutoTokenizer

MOBILEO_REPO = os.environ.get("MOBILEO_PATH", ".")
if MOBILEO_REPO not in sys.path:
    sys.path.insert(0, MOBILEO_REPO)

from mobileo.model import mobileoForInferenceLM
from mobileo.mm_utils import tokenizer_image_token

UND_IMAGE_SIZE = 1024

SYSTEM_PREFIX = (
    "A chat between a user and an artificial intelligence assistant expert "
    "in Gastrointestinal endoscopic images. The assistant is tasked with "
    "detecting hallucinated sentences in a given caption. Hallucination "
    "occurs when the caption is incorrect, misleading, or non-existent "
    "information that is not grounded in the input image or context. "
    "Hallucinations include factual errors, misidentification of anatomy, "
    "false detection of abnormalities, incorrect reasoning, and nonexistent "
    "instruments or conditions like bleeding, infection, or inflammation.\n\n"
)


def load_model(model_path: str, device: str = "cuda"):
    """
    Load the finetuned model, tokenizer, and image processor from a checkpoint.

    Args:
        model_path: Path to the checkpoint directory (e.g. checkpoints/vlm_gutvlm_hal/epoch_4)
        device: "cuda" or "cpu"

    Returns:
        (model, tokenizer, image_processor)
    """
    print(f"[inference] Loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = mobileoForInferenceLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    image_processor = model.get_model().get_vision_tower().image_processor
    print("[inference] Model loaded.")
    return model, tokenizer, image_processor


def _load_and_preprocess_image(image_input: Union[str, Path, "Image.Image"], image_processor, device: str):
    """Load image from path or PIL, resize to 1024x1024, return bfloat16 tensor [1, C, H, W]."""
    if isinstance(image_input, (str, Path)):
        image = Image.open(image_input).convert("RGB")
    else:
        image = image_input.convert("RGB")
    pixel_values = image_processor.preprocess(
        image,
        return_tensors="pt",
        size={"height": UND_IMAGE_SIZE, "width": UND_IMAGE_SIZE},
    )["pixel_values"].to(device, dtype=torch.bfloat16)
    return pixel_values


def _tokenize_prompt(prompt: str, tokenizer) -> torch.LongTensor:
    """
    Tokenize a prompt containing a literal '<image>' placeholder.
    Returns a [1, seq_len] LongTensor with IMAGE_TOKEN_INDEX (-200) at the
    image position, ready for mobileoForInferenceLM.generate(input_ids=...).
    """
    return tokenizer_image_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0)


def ask(
    model,
    tokenizer,
    image_processor,
    image: Union[str, Path, "Image.Image"],
    question: str,
    max_new_tokens: int = 256,
) -> str:
    """
    Single-turn VQA inference.

    Prompt format matches training:
        <|im_start|>user\\n<image>\\n{question}<|im_start|>assistant\\n
    (Human turns close with <|im_start|>assistant rather than <|im_end|> --
    this mirrors exactly how the dataset class built training sequences.)

    Returns the model's response as a plain string.
    """
    device = next(model.parameters()).device
    und_image = _load_and_preprocess_image(image, image_processor, device)
    prompt = f"<|im_start|>user\n<image>\n{question}<|im_start|>assistant\n"
    input_ids = _tokenize_prompt(prompt, tokenizer).to(device)

    output_ids = model.generate(
        input_ids=input_ids,
        images=und_image,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def detect_hallucinations(
    model,
    tokenizer,
    image_processor,
    image: Union[str, Path, "Image.Image"],
    caption: str,
    max_new_tokens: int = 512,
) -> tuple:
    """
    Two-turn hallucination detection (matches Gut-VLM training format).

    Turn 1: model tags each sentence of `caption` as
            <non-hallucinated> or <hallucinated>.
    Turn 2: model produces a corrected version of the caption
            (starting with "Modified caption: ").

    Returns:
        (detection_output, corrected_caption)
        - detection_output: per-sentence tag string
        - corrected_caption: corrected text (prefix "Modified caption: " stripped)
    """
    device = next(model.parameters()).device
    und_image = _load_and_preprocess_image(image, image_processor, device)

    # --- Stage 1: detect hallucinations ---
    detect_prompt = (
        f"<|im_start|>user\n"
        f"{SYSTEM_PREFIX}<image>\nCaption: {caption}\n\n"
        "Can you detect which sentences are hallucinated in the given caption?"
        "<|im_start|>assistant\n"
    )
    detect_input_ids = _tokenize_prompt(detect_prompt, tokenizer).to(device)
    detect_ids = model.generate(
        input_ids=detect_input_ids,
        images=und_image,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    detection = tokenizer.decode(detect_ids[0], skip_special_tokens=True).strip()

    # --- Stage 2: correct hallucinated sentences ---
    # Replicate the full conversation so far, then ask for the correction.
    # <image> appears only once (first human turn) — matches training exactly.
    im_end = "<|im_end|>"
    correct_prompt = (
        f"<|im_start|>user\n"
        f"{SYSTEM_PREFIX}<image>\nCaption: {caption}\n\n"
        "Can you detect which sentences are hallucinated in the given caption?"
        f"<|im_start|>assistant\n{detection}{im_end}\n"
        "<|im_start|>user\n"
        "Can you please correct any hallucinated sentences and generate a "
        "modified response?"
        "<|im_start|>assistant\n"
    )
    correct_input_ids = _tokenize_prompt(correct_prompt, tokenizer).to(device)
    correct_ids = model.generate(
        input_ids=correct_input_ids,
        images=und_image,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    correction = tokenizer.decode(correct_ids[0], skip_special_tokens=True).strip()

    # Strip the "Modified caption: " prefix the model was trained to output
    if correction.lower().startswith("modified caption:"):
        correction = correction[len("Modified caption:"):].strip()

    return detection, correction


def _parse_args():
    p = argparse.ArgumentParser(description="Mobile-O VLM inference")
    p.add_argument("--model_path", required=True,
                   help="Path to finetuned checkpoint (e.g. checkpoints/vlm_gutvlm_hal/epoch_4)")
    p.add_argument("--image", required=True, help="Path to the input image")
    p.add_argument("--mode", choices=["vqa", "hallucination"], default="vqa")
    p.add_argument("--question", default="What do you see in this endoscopy image?",
                   help="Question for VQA mode")
    p.add_argument("--caption", default="",
                   help="Caption to analyse in hallucination mode")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    model, tokenizer, image_processor = load_model(args.model_path, args.device)

    if args.mode == "vqa":
        answer = ask(model, tokenizer, image_processor,
                     args.image, args.question, args.max_new_tokens)
        print("\n=== Answer ===")
        print(answer)
    else:
        if not args.caption:
            raise ValueError("--caption is required for hallucination mode")
        detection, correction = detect_hallucinations(
            model, tokenizer, image_processor,
            args.image, args.caption, args.max_new_tokens,
        )
        print("\n=== Hallucination Detection ===")
        print(detection)
        print("\n=== Corrected Caption ===")
        print(correction)
