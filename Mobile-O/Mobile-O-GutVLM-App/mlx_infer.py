"""
Native macOS inference for the finetuned GutVLM checkpoint, using Core ML for the
vision encoder and MLX for the Qwen2-0.5B LLM. No iPhone, no Xcode, no PyTorch at
inference time -- only the export step (export.py) needs PyTorch.

Mirrors Mobile-O/inference.py's ask()/detect_hallucinations() exactly (same
SYSTEM_PREFIX, same two-turn hallucination prompt format, same <image> sentinel
splice logic), but sourced from the Core ML + MLX artifacts produced by
export.py instead of the original PyTorch checkpoint.

Usage:
    from mlx_infer import load_state, ask, detect_hallucinations
    state = load_state("exported_models")
    answer = ask(state, image, "Is there a polyp visible?")
    detection, correction = detect_hallucinations(state, image, caption)
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import coremltools as ct
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor
from mlx_lm.models import qwen2
from mlx_lm.generate import generate_step

IMAGE_TOKEN_INDEX = -200  # same sentinel convention as mobileo/constants.py
UND_IMAGE_SIZE = 1024

# Copied character-for-character from Mobile-O/inference.py.
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


class MMProjector(nn.Module):
    """mlp2x_gelu: Linear(mm_hidden,hidden) -> GELU -> Linear(hidden,hidden).

    Matches build_vision_projector()'s 'mlp2x_gelu' Sequential in the PyTorch
    model (mm_projector.0 = linear_0, mm_projector.2 = linear_2, index 1 is
    the parameter-free GELU) -- export.py remaps those weight names exactly.
    """

    def __init__(self, mm_hidden_size: int, hidden_size: int):
        super().__init__()
        self.linear_0 = nn.Linear(mm_hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_2(nn.gelu(self.linear_0(x)))


@dataclass
class GutVLMState:
    llm: qwen2.Model
    projector: MMProjector
    tokenizer: object
    vision_model: ct.models.MLModel
    image_processor: CLIPImageProcessor


def load_state(exported_dir: Union[str, Path]) -> GutVLMState:
    """Load the Core ML vision encoder + MLX LLM/projector produced by export.py."""
    exported_dir = Path(exported_dir)
    llm_dir = exported_dir / "llm"
    cfg = json.loads((llm_dir / "config.json").read_text())

    args = qwen2.ModelArgs.from_dict({**cfg, "model_type": "qwen2"})
    llm = qwen2.Model(args)
    projector = MMProjector(cfg["mm_hidden_size"], cfg["hidden_size"])

    quant = cfg.get("quantization")
    if quant:
        nn.quantize(llm, group_size=quant["group_size"], bits=quant["bits"])
        nn.quantize(projector, group_size=quant["group_size"], bits=quant["bits"])

    weights = mx.load(str(llm_dir / "model.safetensors"))
    llm_weights = [
        (k[len("language_model."):], v)
        for k, v in weights.items()
        if k.startswith("language_model.")
    ]
    proj_weights = [
        (k[len("multi_modal_projector."):], v)
        for k, v in weights.items()
        if k.startswith("multi_modal_projector.")
    ]
    llm.load_weights(llm_weights)
    projector.load_weights(proj_weights)
    mx.eval(llm.parameters())
    mx.eval(projector.parameters())

    tokenizer = AutoTokenizer.from_pretrained(str(llm_dir))

    vision_model = ct.models.MLModel(str(exported_dir / "vision_encoder.mlpackage"))
    # Same params as MobileCLIPVisionTower.load_model(): no ImageNet mean/std,
    # just a plain resize + [0,1] scale.
    image_processor = CLIPImageProcessor(
        crop_size={"height": UND_IMAGE_SIZE, "width": UND_IMAGE_SIZE},
        image_mean=[0.0, 0.0, 0.0],
        image_std=[1.0, 1.0, 1.0],
        size={"shortest_edge": UND_IMAGE_SIZE},
    )

    return GutVLMState(llm, projector, tokenizer, vision_model, image_processor)


def _preprocess_image(image_processor: CLIPImageProcessor, image: Union[str, Path, Image.Image]) -> np.ndarray:
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    image = image.convert("RGB")
    pixel_values = image_processor.preprocess(
        image,
        return_tensors="np",
        size={"height": UND_IMAGE_SIZE, "width": UND_IMAGE_SIZE},
    )["pixel_values"]
    return pixel_values.astype(np.float32)  # [1, 3, 1024, 1024]


def _run_vision(vision_model: ct.models.MLModel, pixel_values: np.ndarray) -> mx.array:
    out = vision_model.predict({"images": pixel_values})
    feats = np.asarray(out["image_features"], dtype=np.float32)  # [1, 256, 3072]
    return mx.array(feats).astype(mx.float16)


def _tokenizer_image_token(prompt: str, tokenizer) -> list:
    """Direct port of mobileo/mm_utils.py's tokenizer_image_token(): split the
    prompt on the literal '<image>' substring, tokenize each chunk, and splice
    in the IMAGE_TOKEN_INDEX sentinel at the split point."""
    chunks = [tokenizer(c).input_ids for c in prompt.split("<image>")]

    input_ids = []
    offset = 0
    if chunks and len(chunks[0]) > 0 and chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(chunks[0][0])

    sep = [IMAGE_TOKEN_INDEX] * (offset + 1)
    interleaved = [x for pair in zip(chunks, [sep] * len(chunks)) for x in pair][:-1]
    for x in interleaved:
        input_ids.extend(x[offset:])
    return input_ids


def _build_inputs_embeds(state: GutVLMState, image_features_3072: mx.array, token_ids: list) -> mx.array:
    """Project image features to LLM space, split token_ids on the -200
    sentinel, embed the text spans, and splice
    [text_before, image_embeds, text_after] into one sequence."""
    image_embeds = state.projector(image_features_3072)[0]  # [256, hidden]
    hidden = image_embeds.shape[-1]

    idx = token_ids.index(IMAGE_TOKEN_INDEX)
    before_ids, after_ids = token_ids[:idx], token_ids[idx + 1:]
    embed_tokens = state.llm.model.embed_tokens

    before = embed_tokens(mx.array(before_ids)[None])[0] if before_ids else mx.zeros((0, hidden), dtype=image_embeds.dtype)
    after = embed_tokens(mx.array(after_ids)[None])[0] if after_ids else mx.zeros((0, hidden), dtype=image_embeds.dtype)

    return mx.concatenate([before, image_embeds, after], axis=0)  # [L, hidden]


def _generate(state: GutVLMState, inputs_embeds: mx.array, max_new_tokens: int) -> str:
    """Greedy-decode from spliced embeddings: prefill with input_embeddings
    (mlx_lm's Qwen2 model natively supports this), then continue token-by-token
    through the normal token-id path, reusing the KV cache."""
    eos_id = state.tokenizer.eos_token_id
    out_ids = []
    for token, _ in generate_step(
        prompt=mx.array([], dtype=mx.int32),
        model=state.llm,
        max_tokens=max_new_tokens,
        input_embeddings=inputs_embeds,
    ):
        if token == eos_id:
            break
        out_ids.append(token)
    return state.tokenizer.decode(out_ids, skip_special_tokens=True).strip()


def ask(state: GutVLMState, image: Union[str, Path, Image.Image], question: str, max_new_tokens: int = 256) -> str:
    """Single-turn VQA inference. Mirrors Mobile-O/inference.py's ask()."""
    pixel_values = _preprocess_image(state.image_processor, image)
    image_features = _run_vision(state.vision_model, pixel_values)

    prompt = f"<|im_start|>user\n<image>\n{question}<|im_start|>assistant\n"
    token_ids = _tokenizer_image_token(prompt, state.tokenizer)
    embeds = _build_inputs_embeds(state, image_features, token_ids)
    return _generate(state, embeds, max_new_tokens)


def detect_hallucinations(
    state: GutVLMState,
    image: Union[str, Path, Image.Image],
    caption: str,
    max_new_tokens: int = 512,
) -> tuple:
    """Two-turn hallucination detection + correction. Mirrors
    Mobile-O/inference.py's detect_hallucinations() exactly."""
    pixel_values = _preprocess_image(state.image_processor, image)
    image_features = _run_vision(state.vision_model, pixel_values)

    detect_prompt = (
        f"<|im_start|>user\n"
        f"{SYSTEM_PREFIX}<image>\nCaption: {caption}\n\n"
        "Can you detect which sentences are hallucinated in the given caption?"
        "<|im_start|>assistant\n"
    )
    detect_ids = _tokenizer_image_token(detect_prompt, state.tokenizer)
    detect_embeds = _build_inputs_embeds(state, image_features, detect_ids)
    detection = _generate(state, detect_embeds, max_new_tokens)

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
    correct_embeds = _build_inputs_embeds(state, image_features, correct_ids)
    correction = _generate(state, correct_embeds, max_new_tokens)

    if correction.lower().startswith("modified caption:"):
        correction = correction[len("Modified caption:"):].strip()

    return detection, correction
