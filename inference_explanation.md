# Inference and Gradio Demo — Detailed Explanation

This document explains exactly how inference works with our finetuned Mobile-O checkpoint,
what every function does, how tensors flow through the system, and why each decision
was made the way it was. Nothing is skipped or summarized.

---

## 1. The Big Picture — What Happens When You Click "Detect & Correct"

When you upload an image and click a button in Gradio, here is the complete sequence
of things that happen, from start to finish:

1. Gradio captures the image (as a PIL Image object) and the caption text.
2. `app.py` calls `detect_hallucinations(model, tokenizer, image_processor, image, caption)` from `inference.py`.
3. Inside that function, the image is resized and turned into a number tensor.
4. The text prompt is tokenized into a list of integer IDs, with a special `-200` placeholder where `<image>` was.
5. The model's `generate()` method is called. That method:
   a. Sends the image through the vision tower to get 256 image feature vectors.
   b. Splices those 256 vectors into the middle of the text embedding sequence (at the `-200` position).
   c. Feeds this combined sequence into the Qwen2 language model, which generates tokens one at a time.
6. The generated tokens are decoded back to text — that becomes the detection output.
7. Steps 3–6 are repeated for the correction turn, this time feeding the detection response
   back as part of the conversation context.
8. Gradio displays both outputs in the text boxes.

Everything below expands each step in full detail.

---

## 2. The Checkpoint — What is Actually Saved in `epoch_4/`

The checkpoint at `checkpoints/vlm_gutvlm_hal/epoch_4/` is a folder containing:

```
epoch_4/
├── config.json              ← model architecture config (hidden_size, num_layers, etc.)
├── model.safetensors        ← the actual learned weights (several GB)
├── tokenizer.json           ← tokenizer vocabulary and rules
├── tokenizer_config.json    ← tokenizer settings
├── special_tokens_map.json  ← maps special tokens like <|im_start|>, <|im_end|>
└── ...
```

`model.safetensors` contains the weights for the entire Mobile-O architecture:

| Component | Parameters | Trained? |
|---|---|---|
| vision_tower (FastViT MobileCLIP) | ~600M | Yes (Step 2 Kvasir-VQA, then frozen in Step 3) |
| mm_projector (2-layer MLP) | ~4M | Yes |
| LLM (Qwen2-0.5B) | ~500M | Yes |
| dit (SANA diffusion transformer) | ~548 layers | No — frozen, not used for inference |
| sana_vae (image VAE) | large | No — frozen, not used for inference |
| diffusion_connector | 26 layers | No — frozen, not used for inference |

So the checkpoint contains the frozen generation half too (DiT, VAE, diffusion_connector).
These weights are loaded into memory but never called during text inference. They just sit
there. The inference path only activates: **vision_tower → mm_projector → LLM**.

---

## 3. Loading the Model — Why We Cannot Use `AutoModelForCausalLM`

Standard HuggingFace loading would be:
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("checkpoints/vlm_gutvlm_hal/epoch_4")
```

This **fails** with `KeyError: 'mobile_o_inference'`. The reason: HuggingFace's AutoModel
system reads `config.json` and finds `"model_type": "mobile_o_inference"`. It looks up
that string in its internal registry. Mobile-O is not an official HuggingFace model, so
the registry doesn't have it, and it crashes.

The fix is to use Mobile-O's own class, which registers itself:

```python
# At the bottom of mobileo_inference.py:
AutoConfig.register("mobileo_inference", mobileoConfig)
AutoModelForCausalLM.register(mobileoConfig, mobileoForInferenceLM)
```

This line runs when you do `from mobileo.model import mobileoForInferenceLM`, which
registers the class into HuggingFace's system. After that, both the custom class and
`AutoModelForCausalLM` would work. But we use the custom class directly to be explicit.

In `inference.py`, `load_model()` does:
```python
from mobileo.model import mobileoForInferenceLM

model = mobileoForInferenceLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
```

- `torch_dtype=torch.bfloat16`: load weights as 16-bit brain float instead of 32-bit.
  This halves memory usage. bfloat16 has the same exponent range as float32 but fewer
  mantissa bits — good enough for inference, and what the model was trained in.
- `low_cpu_mem_usage=True`: stream the checkpoint from disk rather than loading it all
  into RAM at once, then copying to GPU. Avoids doubling RAM usage during loading.

After loading, `model.to(device)` moves all weights to the GPU (CUDA).

---

## 4. The Model Class — `mobileoForInferenceLM` Inheritance Chain

```
mobileoForInferenceLM
    ├── Qwen2ForCausalLM        ← provides: embed_tokens, transformer layers, lm_head, generate()
    └── LlavaMetaForCausalLM   ← provides: prepare_inputs_labels_for_multimodal(), visual()
```

The class is defined in `mobileo/model/language_model/mobileo_inference.py`:

```python
class mobileoForInferenceLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = mobileoConfig

    def __init__(self, config):
        super().__init__(config)
        config.model_type = "mobileo_inference"
        config.is_train = False
        self.model = mobileoModel(config)        # contains all submodules
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        return self.model                        # shortcut to access submodules
```

`self.model` (of type `mobileoModel`) contains:
- `self.model.vision_tower` — the FastViT MobileCLIP encoder
- `self.model.mm_projector` — the MLP that maps vision features to LLM's hidden size
- `self.model.embed_tokens` — the LLM's token embedding table
- `self.model.layers` — the 24 transformer decoder layers of Qwen2-0.5B
- `self.model.dit`, `self.model.sana_vae`, etc. — the generation half (unused in inference)

`lm_head` is a linear layer that maps from the LLM's 896-dimensional hidden state to
vocabulary logits (vocab size ~152,064 for Qwen2).

---

## 5. The Image Processor — Turning a JPEG into a Tensor

The image processor lives inside the vision tower:

```python
image_processor = model.get_model().get_vision_tower().image_processor
```

It is a standard HuggingFace `CLIPImageProcessor` (or similar) that was set during
Mobile-O's original pretraining. We load it from the checkpoint, not from scratch.

In `inference.py`, `_load_and_preprocess_image()` does:
```python
pixel_values = image_processor.preprocess(
    image,
    return_tensors="pt",
    size={"height": 1024, "width": 1024},
)["pixel_values"].to(device, dtype=torch.bfloat16)
```

What happens inside `preprocess()`:
1. Resize the image to exactly 1024×1024 pixels (regardless of original aspect ratio).
2. Normalize each pixel: subtract the mean `[0.485, 0.456, 0.406]` and divide by
   std `[0.229, 0.224, 0.225]` (ImageNet statistics — the vision tower was pretrained
   with these).
3. Convert from HWC (Height × Width × Channels) to CHW format (Channels first).
4. Return a PyTorch tensor.

Output shape: `[1, 3, 1024, 1024]` — batch of 1, RGB, 1024 tall, 1024 wide.
Dtype: `torch.bfloat16`.

The reason for 1024×1024 (not the more common 224×224 or 336×336): Mobile-O's
MobileCLIP / FastViT vision encoder was designed for high-resolution inputs to capture
fine-grained medical details. This is set as `UND_IMAGE_SIZE = 1024` in both the
training script and `inference.py`.

---

## 6. Tokenization — Turning the Prompt into Integer IDs

Standard HuggingFace tokenizers know nothing about `<image>` as an image placeholder.
Mobile-O uses a special function `tokenizer_image_token()` from `mobileo/mm_utils.py`:

```python
def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])
    ...
```

What this does, step by step:

1. Split the prompt string on the literal text `"<image>"`.
   Example prompt: `"<|im_start|>user\n<image>\nWhat do you see?<|im_start|>assistant\n"`
   After split: `["<|im_start|>user\n", "\nWhat do you see?<|im_start|>assistant\n"]`

2. Tokenize each chunk independently using the Qwen2 tokenizer.
   Chunk 0 → `[151644, 882, 198]`  (tokens for `<|im_start|>`, `user`, newline)
   Chunk 1 → `[198, 3838, 653, 498, 1518, 30, 151644, 77091, 198]`  (tokens for the rest)

3. Insert `IMAGE_TOKEN_INDEX = -200` between the chunks.
   Result: `[151644, 882, 198, -200, 198, 3838, 653, 498, 1518, 30, 151644, 77091, 198]`

4. Return as `torch.tensor([...], dtype=torch.long)`, then `.unsqueeze(0)` gives shape `[1, seq_len]`.

The value `-200` is not a real vocabulary token (Qwen2 vocabulary uses IDs 0 to ~152,063).
It is a sentinel that later code (`prepare_inputs_labels_for_multimodal`) searches for and
replaces with actual image feature vectors. This is how the system marks "insert image here".

`IMAGE_TOKEN_INDEX = -200` is defined in `mobileo/constants.py`.

---

## 7. The `generate()` Override — The Core of Inference

This is the most important function to understand. It lives in `mobileoForInferenceLM`:

```python
@torch.no_grad()
def generate(self, input_ids=None, images=None, **kwargs):
    self.to(torch.float32)                          # (A) cast everything to float32
    position_ids = kwargs.pop("position_ids", None)
    attention_mask = kwargs.pop("attention_mask", None)
    if "inputs_embeds" in kwargs:
        raise NotImplementedError("`inputs_embeds` is not supported")  # (B)

    if images is not None:
        (input_ids, position_ids, attention_mask, _,
         inputs_embeds, _, _) = self.prepare_inputs_labels_for_multimodal(
            input_ids, position_ids, attention_mask, None, None, und_images=images,
        )                                           # (C) inject image features
    else:
        inputs_embeds = self.get_model().embed_tokens(input_ids)   # text-only fallback

    return super().generate(                        # (D) Qwen2ForCausalLM.generate()
        position_ids=position_ids,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        **kwargs
    )
```

**(A) `self.to(torch.float32)`:**
This converts the entire model — all 1.5+ billion parameters — from bfloat16 to float32
before generating. The original Mobile-O authors added this. The reason: certain operations
(softmax, layer norm, attention score computation) can overflow or underflow in bfloat16's
limited mantissa range, producing slightly wrong probabilities that compound token by token.
float32 gives stable, deterministic generation. The downside is doubled GPU memory for the
duration of generation, but on a 96GB GH200 that is not a problem.

**(B) `inputs_embeds` is not supported:**
If you tried to call `model.generate(inputs_embeds=some_tensor)`, this line would raise
an error. This is why `inference.py` cannot do that — it must pass `input_ids` and
`images` and let the generate() override handle the injection internally.

**(C) `prepare_inputs_labels_for_multimodal()` — see Section 8 below for full detail.**
This is where the image gets "injected" into the token sequence. After this call,
`input_ids` becomes `None` (no longer needed) and `inputs_embeds` is a tensor of shape
`[1, expanded_seq_len, 896]` where `expanded_seq_len = original_seq_len - 1 + 256`
(one `-200` placeholder replaced by 256 image feature vectors).

**(D) `super().generate()` = `Qwen2ForCausalLM.generate()`:**
This is the standard HuggingFace generate loop. It receives `inputs_embeds` (not
`input_ids`) for the first step — the entire sequence including image features. It runs
the Qwen2 transformer forward pass, gets logits for the next token, samples or takes
argmax, then appends that token and runs forward again. On each step after the first,
it passes `input_ids=[1,1]` (just the new token) and uses the KV cache — it does NOT
re-process the image again. The generate loop continues until it produces `eos_token_id`
(`<|im_end|>`) or hits `max_new_tokens`.

Output: `output_ids` — a LongTensor of shape `[1, num_generated_tokens]` containing
just the newly generated token IDs (not the prompt, because the prompt was passed as
embeddings, not IDs).

---

## 8. `prepare_inputs_labels_for_multimodal()` — The Image Injection Engine

This function is defined in `mobileo/model/llava_arch.py` and is the heart of the
multimodal system. Here is what it does, line by line:

### 8.1 The Early Return for Autoregressive Steps

```python
if (gen_images is None and und_images is None) or input_ids.shape[1] == 1 or self.get_vision_tower() is None:
    return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None, None
```

During autoregressive generation, after the first token, Qwen2's generate loop calls
`model.forward(input_ids=[[new_token_id]], past_key_values=kv_cache)`. Here
`input_ids.shape[1] == 1`. This condition fires, and the function returns immediately
without touching the image. The single new token goes through `embed_tokens()` normally.
This is correct and intentional — the image was already processed in the first forward
pass and its KV cache entries are stored. The model does not "see" the image again on
subsequent tokens; it uses the cached attention states instead.

### 8.2 Running the Vision Tower

```python
image_features = self.visual(images)   # images shape: [1, 3, 1024, 1024]
```

`self.visual()` is a method on `LlavaMetaForCausalLM` that runs:
1. `vision_tower(images)` — the FastViT MobileCLIP encoder processes the 1024×1024 image.
   FastViT divides the image into patches, runs them through its transformer, and produces
   a feature map. The output is `[1, 256, 256]` — batch 1, 256 image tokens, 256-dim features.
   (The 256 image tokens come from the number of spatial patches the vision tower produces.)
2. `mm_projector(features)` — the 2-layer MLP maps from vision space (256-dim) to LLM space (896-dim).
   Output: `[1, 256, 896]`.

So `image_features` has shape `[1, 256, 896]`. This is 256 vectors, each 896-dimensional,
representing the visual content of the image in the same embedding space as the LLM's word tokens.

### 8.3 Splitting the Text Around the Image Placeholder

```python
image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
```

This finds the position of `-200` in `input_ids`. For a prompt with one image, there is
exactly one position. The code builds a list of text chunk boundaries:
- `[-1, position_of_minus_200, end_of_sequence]`

Then it extracts two text chunks:
- `text_before_image`: tokens from index 0 to `position_of_minus_200 - 1`
- `text_after_image`: tokens from `position_of_minus_200 + 1` to end

### 8.4 Embedding the Text Chunks

```python
cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
```

The token IDs for the text parts are converted to embedding vectors using Qwen2's
`embed_tokens` lookup table (size: `[152064, 896]` — 152,064 vocabulary entries, each
896-dimensional). Each integer token ID becomes a 896-dimensional vector.

### 8.5 Concatenating Text + Image + Text

```python
cur_new_input_embeds = torch.cat([
    text_before_image_embeds,    # shape [n_before, 896]
    image_features[0],           # shape [256, 896]
    text_after_image_embeds,     # shape [n_after, 896]
], dim=0)
```

The result is a single sequence of embeddings:
- All tokens before `<image>` (as word embeddings)
- 256 image feature vectors (from vision tower + mm_projector)
- All tokens after `<image>` (as word embeddings)

Total length: `n_before + 256 + n_after = original_seq_len - 1 + 256`.

If the original tokenized prompt had 40 tokens with one `-200` placeholder, the output
embedding sequence has `40 - 1 + 256 = 295` vectors, each 896-dimensional.

The `attention_mask` and `position_ids` are also expanded to match this new length.

### 8.6 Return Value

```python
return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, target_image_embeds
```

- First value is `None` — `input_ids` is discarded because the text is now embedded.
- 5th value `new_input_embeds` shape: `[1, expanded_seq_len, 896]` — this is what gets
  passed to `super().generate()`.
- Labels return value: during inference we pass `labels=None`, so it also returns `None` here.

---

## 9. Prompt Format — Why It Looks Like It Does

### 9.1 Training Format (from `step3_finetune_hallucination.py`)

The dataset class built training sequences by concatenating turns like this:

```
<|im_start|>user\n{human1_value}<|im_start|>assistant\n{gpt1_value}<|im_end|>\n
<|im_start|>user\n{human2_value}<|im_start|>assistant\n{gpt2_value}<|im_end|>\n
```

Note what is **absent**: there is no `<|im_end|>` after the human/user turns. The
dataset class only appended `<|im_end|>\n` after assistant turns:

```python
if msg["from"] == "human":
    labels.extend([IGNORE_INDEX] * len(msg_ids))   # no <|im_end|> added
else:
    labels.extend(msg_ids)
    tokens.append(im_end_id)   # <|im_end|> only after assistant
    tokens.append(eol_id)      # newline after assistant
```

During inference the prompt must match this format exactly, because the model learned
to generate after seeing `<|im_start|>assistant\n` — not after `<|im_end|>\n<|im_start|>assistant\n`.
If you use the wrong format, the model's generation distribution shifts and outputs
degrade.

### 9.2 VQA Mode Prompt

```
<|im_start|>user\n<image>\n{question}<|im_start|>assistant\n
```

The model sees: [user role marker] [image: 256 feature vectors] [question tokens] [assistant role marker]
Then it generates the answer.

### 9.3 Hallucination Detection — Turn 1 Prompt

```
<|im_start|>user\n
{SYSTEM_PREFIX}<image>\nCaption: {original_caption}\n\nCan you detect which sentences are hallucinated in the given caption?
<|im_start|>assistant\n
```

`SYSTEM_PREFIX` is a 6-sentence paragraph explaining that the model is a GI expert
detecting hallucinations. This was the prefix used in training data
(from `step3_gut_vlm_data.py`). It must be reproduced exactly here — the model was
trained to respond with per-sentence tags when it sees this prefix. Without it, the
model would not know to produce `<hallucinated>` / `<non-hallucinated>` tags.

### 9.4 Hallucination Detection — Turn 2 Prompt (Correction)

```
<|im_start|>user\n
{SYSTEM_PREFIX}<image>\nCaption: {original_caption}\n\nCan you detect...?
<|im_start|>assistant\n
{detection_output}<|im_end|>\n
<|im_start|>user\n
Can you please correct any hallucinated sentences and generate a modified response?
<|im_start|>assistant\n
```

Key point: `<image>` still appears **only once**, in the first human turn. The second
human turn ("Can you please correct...") has no `<image>`. When this full string is
passed to `tokenizer_image_token()`, it finds exactly one `-200` placeholder, and
`prepare_inputs_labels_for_multimodal()` inserts exactly one set of 256 image features —
at the correct position. This exactly mirrors the training data format from
`step3_gut_vlm_data.py`.

The detection output from Turn 1 is included verbatim as context so the model knows
what it already said and can produce a consistent correction.

---

## 10. What Changed from the Original Mobile-O Inference

The original Mobile-O repository was designed for both **understanding** (text from image)
and **generation** (image from text). Its `generate()` in `mobileoForInferenceLM` was
written for text generation from images. We did not change this function at all.

What we changed and why:

### 10.1 The Training Wrapper Class (`mobileoUnderstandingForTraining`)

During training (`step3_finetune_hallucination.py`), we could not use the original
`mobileoFastSFTForCausalLM` because its `forward()` has:

```python
assert latents is not None
```

This assert runs diffusion loss unconditionally — it assumes you always have a target
image to generate. We only want the language modeling loss (predicting text tokens),
not diffusion loss. So we subclassed `mobileoForInferenceLM` and wrote our own
`forward()` that calls `prepare_inputs_labels_for_multimodal()` and then
`super(mobileoForInferenceLM, self).forward()` (which is `Qwen2ForCausalLM.forward()`
— no diffusion assert).

**At inference time, this custom forward() is not used.** The model is loaded as
`mobileoForInferenceLM` (the base class), and `generate()` calls `Qwen2ForCausalLM.generate()`
which internally calls `Qwen2ForCausalLM.forward()` — not our training wrapper's forward.
The finetuned weights are all there; the class used to load them does not matter as long
as the architecture matches.

### 10.2 The Frozen Vision Tower

During training (Step 3), we froze the vision tower:
```python
for p in model.get_model().get_vision_tower().parameters():
    p.requires_grad = False
model.get_model().get_vision_tower().eval()
```

At inference time, the vision tower is always in eval mode anyway (`model.eval()`
freezes all batch norm statistics). Its weights in `epoch_4/` are identical to what
they were in `vlm_kvasir_full_continued/epoch_2/` — the Step 2 Kvasir-VQA trained
vision tower. The LLM and mm_projector weights are the ones that changed during Step 3.

### 10.3 What We Did NOT Change

- `prepare_inputs_labels_for_multimodal()` in `llava_arch.py` — untouched.
- `generate()` in `mobileoForInferenceLM` — untouched.
- `tokenizer_image_token()` in `mm_utils.py` — untouched.
- The checkpoint format — we used `save_pretrained()` which saves in the same format
  as the original, so `from_pretrained()` loads it identically.

---

## 11. How `inference.py` Uses These Pieces

### `load_model(model_path, device)`

```python
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = mobileoForInferenceLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device)
model.eval()
image_processor = model.get_model().get_vision_tower().image_processor
return model, tokenizer, image_processor
```

- Loads tokenizer from the same checkpoint folder (it contains the Qwen2 tokenizer files).
- Loads model weights into bfloat16 on CPU first, then moves to GPU.
- `model.eval()` disables dropout and sets BatchNorm to use running statistics (not batch stats).
- Extracts the image processor from inside the loaded vision tower — this ensures the
  processor settings exactly match what was used during training.

### `ask(model, tokenizer, image_processor, image, question, max_new_tokens)`

```python
prompt = f"<|im_start|>user\n<image>\n{question}<|im_start|>assistant\n"
input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0).to(device)
und_image = _load_and_preprocess_image(image, image_processor, device)

output_ids = model.generate(
    input_ids=input_ids,
    images=und_image,
    max_new_tokens=max_new_tokens,
    do_sample=False,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
)
return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
```

- `do_sample=False` means greedy decoding — always pick the highest-probability next token.
  No randomness. This gives deterministic, repeatable outputs.
- `eos_token_id=tokenizer.eos_token_id` — stop generating when the model produces the
  end-of-sequence token (which for Qwen2 is `<|im_end|>`).
- `output_ids[0]` — batch dimension unwrapped (we always run batch size 1).
- `skip_special_tokens=True` — removes `<|im_start|>`, `<|im_end|>`, etc. from the decoded string.

### `detect_hallucinations(model, tokenizer, image_processor, image, caption, max_new_tokens)`

This makes **two separate `model.generate()` calls**:

**Call 1 (detection):**
```python
detect_prompt = f"<|im_start|>user\n{SYSTEM_PREFIX}<image>\nCaption: {caption}\n\nCan you detect which sentences are hallucinated in the given caption?<|im_start|>assistant\n"
detect_input_ids = tokenizer_image_token(detect_prompt, tokenizer, return_tensors="pt").unsqueeze(0).to(device)
detect_ids = model.generate(input_ids=detect_input_ids, images=und_image, ...)
detection = tokenizer.decode(detect_ids[0], skip_special_tokens=True).strip()
```

**Call 2 (correction):**
```python
im_end = "<|im_end|>"
correct_prompt = (
    f"<|im_start|>user\n{SYSTEM_PREFIX}<image>\nCaption: {caption}\n\nCan you detect...?"
    f"<|im_start|>assistant\n{detection}{im_end}\n"         # includes detection from call 1
    "<|im_start|>user\nCan you please correct any hallucinated sentences and generate a modified response?"
    "<|im_start|>assistant\n"
)
correct_input_ids = tokenizer_image_token(correct_prompt, tokenizer, return_tensors="pt").unsqueeze(0).to(device)
correct_ids = model.generate(input_ids=correct_input_ids, images=und_image, ...)
correction = tokenizer.decode(correct_ids[0], skip_special_tokens=True).strip()
```

Each call processes the image independently (vision tower runs twice). This is slightly
redundant but clean — the alternative would be to cache the KV states from call 1 and
continue, which HuggingFace supports but adds complexity. Since the model is fast and the
images are not large, running the image twice adds only ~100ms on a GH200.

The `detection` string from call 1 is inserted into the prompt for call 2. This gives
the model the full conversational context it was trained on — it learned to produce
corrections after seeing its own detection output. Without including the detection in
prompt 2, the correction quality would degrade because the model trained on seeing what
it said in turn 2 before generating turn 4.

---

## 12. How `app.py` (Gradio) Wraps Everything

`app.py` is a thin UI layer. It does nothing intelligent — it just calls functions from
`inference.py` and puts the results in text boxes.

```python
from inference import load_model, ask, detect_hallucinations

model, tokenizer, image_processor = load_model(args.model_path, args.device)
demo = build_demo(model, tokenizer, image_processor)
demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
```

The model is loaded **once** at startup and held in GPU memory for the entire session.
Each time you click a button, Gradio calls `vqa_fn` or `hallucination_fn`:

```python
def vqa_fn(image, question, max_tokens):
    return ask(model, tokenizer, image_processor, image, question, int(max_tokens))

def hallucination_fn(image, caption, max_tokens):
    detection, correction = detect_hallucinations(
        model, tokenizer, image_processor, image, caption, int(max_tokens)
    )
    return detection, correction
```

`image` here is a `PIL.Image` object — Gradio automatically decodes the uploaded JPEG/PNG.
`inference.py`'s `_load_and_preprocess_image()` handles both PIL Images and file paths,
so this works without any conversion.

`demo.launch(server_name="0.0.0.0")` binds to all network interfaces (not just localhost),
which is required on Clariden to accept port-forwarded connections from your laptop.

`share=True` additionally creates a tunnel to Gradio's public servers and prints a URL like
`https://abc123.gradio.live` that anyone can open — no SSH needed.

---

## 13. Tensor Shape Summary — End to End

| Step | Operation | Input Shape | Output Shape |
|---|---|---|---|
| Image load | PIL → numpy | (H, W, 3) | (H, W, 3) |
| Preprocess | resize + normalize | (H, W, 3) | [1, 3, 1024, 1024] |
| Tokenize | tokenizer_image_token | string | [1, seq_len] (with -200) |
| Vision tower | FastViT forward | [1, 3, 1024, 1024] | [1, 256, 256] |
| mm_projector | 2-layer MLP | [1, 256, 256] | [1, 256, 896] |
| embed_tokens | vocab lookup | [1, seq_len-1] (text only) | [1, seq_len-1, 896] |
| Concatenate | cat at image position | [n_before, 896] + [256, 896] + [n_after, 896] | [1, seq_len-1+256, 896] |
| LLM forward | 24 transformer layers | [1, expanded_len, 896] | [1, expanded_len, 896] |
| lm_head | linear | [1, 1, 896] | [1, 1, 152064] |
| argmax | greedy decode | [1, 1, 152064] | scalar token_id |
| decode | tokenizer | [1, n_generated] | string |

The key expansion: `seq_len - 1 + 256`. One `-200` placeholder disappears and 256 image
feature vectors appear in its place. This is why the LLM "sees" the image — its attention
mechanism processes image and text tokens in the same sequence, at the same positions.

---

## 14. Why This Works for Hallucination Detection Specifically

The model was trained on 1,450 examples where:
- Input: system prompt + image + an AI-generated caption that contains some hallucinated sentences
- Output turn 1: each sentence labeled `<hallucinated>` or `<non-hallucinated>`
- Output turn 2: a corrected version of the caption

After 6 epochs of gradient descent on this data, the LLM's weights learned:
- The visual patterns in Kvasir-v2 endoscopy images (from both Step 2 and the frozen vision tower)
- The mapping from (image + caption sentence) → hallucination label
- The mapping from (image + labelled sentences) → corrected text

At inference time, the model is doing pattern completion: given a new image + caption
it has never seen, it continues the conversation pattern it learned, applying the
hallucination detection logic it was trained on.

The val_loss of 0.1392 at epoch_4 means the model assigns relatively high probability to
the correct next tokens on held-out examples — it has genuinely learned the task, not
just memorized training examples.
