"""
Cross-platform inference for the finetuned GutVLM checkpoint using ONNX Runtime.
Runs on any laptop with a CPU (Windows / Linux / Intel or Apple Mac) -- no CoreML,
no MLX, no PyTorch at inference time. Only onnx_export.py needs PyTorch.

Mirrors Mobile-O/inference.py and mlx_infer.py exactly (same SYSTEM_PREFIX, same
two-turn hallucination prompt format, same <image> sentinel splice logic), but
sourced from the three ONNX graphs produced by onnx_export.py:

    vision_encoder.onnx   pixel_values -> projected image embeds [1,256,896]
    embed_tokens.onnx     input_ids    -> text embeds [1,n,896]
    decoder.onnx          inputs_embeds + position_ids + past_kv -> logits + present_kv

Usage:
    from onnx_infer import load_state, ask, detect_hallucinations
    state = load_state("onnx_models")
    print(ask(state, "img.jpg", "Is there a polyp visible?"))

    # or from the command line:
    python onnx_infer.py --onnx-dir onnx_models --image img.jpg --question "..."
    python onnx_infer.py --onnx-dir onnx_models --image img.jpg --mode hallucination \
        --caption "The image shows a large polyp with active bleeding."
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor

IMAGE_TOKEN_INDEX = -200        # same sentinel convention as mobileo/constants.py
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


@dataclass
class GutVLMState:
    vision: ort.InferenceSession
    embed: ort.InferenceSession
    decoder: ort.InferenceSession
    tokenizer: object
    image_processor: CLIPImageProcessor
    cfg: dict


def _sess(path: Path, threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        so.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


def load_state(onnx_dir: Union[str, Path], threads: int = 0) -> GutVLMState:
    onnx_dir = Path(onnx_dir)
    cfg = json.loads((onnx_dir / "runtime_config.json").read_text())
    vision = _sess(onnx_dir / "vision_encoder.onnx", threads)
    embed = _sess(onnx_dir / "embed_tokens.onnx", threads)
    decoder = _sess(onnx_dir / "decoder.onnx", threads)
    tokenizer = AutoTokenizer.from_pretrained(str(onnx_dir))
    # Same preprocessing as MobileCLIPVisionTower.load_model(): plain resize +
    # [0,1] scale, NO ImageNet mean/std.
    image_processor = CLIPImageProcessor(
        crop_size={"height": UND_IMAGE_SIZE, "width": UND_IMAGE_SIZE},
        image_mean=[0.0, 0.0, 0.0],
        image_std=[1.0, 1.0, 1.0],
        size={"shortest_edge": UND_IMAGE_SIZE},
    )
    return GutVLMState(vision, embed, decoder, tokenizer, image_processor, cfg)


# --------------------------------------------------------------------------- #
#  Preprocessing / component calls                                            #
# --------------------------------------------------------------------------- #

def _preprocess_image(state: GutVLMState, image: Union[str, Path, Image.Image]) -> np.ndarray:
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    image = image.convert("RGB")
    pv = state.image_processor.preprocess(
        image, return_tensors="np",
        size={"height": UND_IMAGE_SIZE, "width": UND_IMAGE_SIZE},
    )["pixel_values"]
    return pv.astype(np.float32)                       # [1, 3, 1024, 1024]


def _run_vision(state: GutVLMState, pixel_values: np.ndarray) -> np.ndarray:
    out = state.vision.run(["image_embeds"], {"pixel_values": pixel_values})[0]
    return out.astype(np.float32)                      # [1, 256, 896]


def _embed(state: GutVLMState, ids: list) -> np.ndarray:
    arr = np.asarray(ids, dtype=np.int64)[None]
    return state.embed.run(["text_embeds"], {"input_ids": arr})[0].astype(np.float32)


def _tokenizer_image_token(prompt: str, tokenizer) -> list:
    """Direct port of mobileo/mm_utils.py's tokenizer_image_token()."""
    chunks = [tokenizer(c).input_ids for c in prompt.split("<image>")]
    input_ids, offset = [], 0
    if chunks and len(chunks[0]) > 0 and chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(chunks[0][0])
    sep = [IMAGE_TOKEN_INDEX] * (offset + 1)
    interleaved = [x for pair in zip(chunks, [sep] * len(chunks)) for x in pair][:-1]
    for x in interleaved:
        input_ids.extend(x[offset:])
    return input_ids


def _build_inputs_embeds(state: GutVLMState, image_embeds: np.ndarray, token_ids: list) -> np.ndarray:
    """Split token_ids on the -200 sentinel, embed the text spans, and splice
    [text_before, image_embeds, text_after] into one [1, L, hidden] sequence."""
    idx = token_ids.index(IMAGE_TOKEN_INDEX)
    before_ids, after_ids = token_ids[:idx], token_ids[idx + 1:]
    hidden = image_embeds.shape[-1]

    before = _embed(state, before_ids) if before_ids else np.zeros((1, 0, hidden), np.float32)
    after = _embed(state, after_ids) if after_ids else np.zeros((1, 0, hidden), np.float32)
    return np.concatenate([before, image_embeds, after], axis=1)   # [1, L, hidden]


# --------------------------------------------------------------------------- #
#  Autoregressive greedy decode with KV cache                                 #
# --------------------------------------------------------------------------- #

def _empty_past(state: GutVLMState) -> dict:
    n_layers = state.cfg["num_hidden_layers"]
    n_kv = state.cfg["num_key_value_heads"]
    hd = state.cfg["hidden_size"] // state.cfg["num_attention_heads"]
    past = {}
    for i in range(n_layers):
        past[f"past_key_{i}"] = np.zeros((1, n_kv, 0, hd), np.float32)
        past[f"past_value_{i}"] = np.zeros((1, n_kv, 0, hd), np.float32)
    return past


def _decode_names(state: GutVLMState):
    n = state.cfg["num_hidden_layers"]
    out = ["logits"]
    for i in range(n):
        out += [f"present_key_{i}", f"present_value_{i}"]
    return out


def _generate(state: GutVLMState, inputs_embeds: np.ndarray, max_new_tokens: int) -> str:
    eos_id = state.cfg["eos_token_id"]
    n_layers = state.cfg["num_hidden_layers"]
    out_names = _decode_names(state)

    L = inputs_embeds.shape[1]
    past = _empty_past(state)
    position_ids = np.arange(L, dtype=np.int64)[None]

    feeds = {"inputs_embeds": inputs_embeds, "position_ids": position_ids, **past}
    outs = state.decoder.run(out_names, feeds)
    logits = outs[0]
    present = outs[1:]
    cur_len = L

    out_ids = []
    for _ in range(max_new_tokens):
        next_id = int(logits[0, -1].argmax())
        if next_id == eos_id:
            break
        out_ids.append(next_id)

        step_embed = _embed(state, [next_id])                       # [1,1,hidden]
        position_ids = np.array([[cur_len]], dtype=np.int64)
        feeds = {"inputs_embeds": step_embed, "position_ids": position_ids}
        for i in range(n_layers):
            feeds[f"past_key_{i}"] = present[2 * i]
            feeds[f"past_value_{i}"] = present[2 * i + 1]
        outs = state.decoder.run(out_names, feeds)
        logits = outs[0]
        present = outs[1:]
        cur_len += 1

    return state.tokenizer.decode(out_ids, skip_special_tokens=True).strip()


# --------------------------------------------------------------------------- #
#  Public API (mirrors inference.py / mlx_infer.py)                           #
# --------------------------------------------------------------------------- #

def ask(state: GutVLMState, image, question: str, max_new_tokens: int = 256) -> str:
    image_embeds = _run_vision(state, _preprocess_image(state, image))
    prompt = f"<|im_start|>user\n<image>\n{question}<|im_start|>assistant\n"
    token_ids = _tokenizer_image_token(prompt, state.tokenizer)
    embeds = _build_inputs_embeds(state, image_embeds, token_ids)
    return _generate(state, embeds, max_new_tokens)


def detect_hallucinations(state: GutVLMState, image, caption: str, max_new_tokens: int = 512) -> tuple:
    image_embeds = _run_vision(state, _preprocess_image(state, image))

    detect_prompt = (
        f"<|im_start|>user\n"
        f"{SYSTEM_PREFIX}<image>\nCaption: {caption}\n\n"
        "Can you detect which sentences are hallucinated in the given caption?"
        "<|im_start|>assistant\n"
    )
    detect_ids = _tokenizer_image_token(detect_prompt, state.tokenizer)
    detection = _generate(state, _build_inputs_embeds(state, image_embeds, detect_ids), max_new_tokens)

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
    correct_ids = _tokenizer_image_token(correct_prompt, state.tokenizer)
    correction = _generate(state, _build_inputs_embeds(state, image_embeds, correct_ids), max_new_tokens)

    if correction.lower().startswith("modified caption:"):
        correction = correction[len("Modified caption:"):].strip()
    return detection, correction


def _parse_args():
    p = argparse.ArgumentParser(description="GutVLM ONNX inference (cross-platform CPU)")
    p.add_argument("--onnx-dir", default="onnx_models")
    p.add_argument("--image", required=True)
    p.add_argument("--mode", choices=["vqa", "hallucination"], default="vqa")
    p.add_argument("--question", default="What do you see in this endoscopy image?")
    p.add_argument("--caption", default="")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--threads", type=int, default=0, help="intra-op threads (0=auto)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    state = load_state(args.onnx_dir, threads=args.threads)
    if args.mode == "vqa":
        print("\n=== Answer ===")
        print(ask(state, args.image, args.question, args.max_new_tokens))
    else:
        if not args.caption:
            raise ValueError("--caption is required for hallucination mode")
        detection, correction = detect_hallucinations(state, args.image, args.caption, args.max_new_tokens)
        print("\n=== Hallucination Detection ===")
        print(detection)
        print("\n=== Corrected Caption ===")
        print(correction)
