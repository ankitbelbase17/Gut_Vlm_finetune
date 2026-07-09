# GutVLM iOS — Finetuned Mobile-O for GI VQA & Hallucination Detection

An on-device iOS app that runs **our finetuned Mobile-O (understanding half)** for
two gastrointestinal-endoscopy tasks:

- **Visual QA** — ask a question about an endoscopy image, get an answer.
- **Hallucination-aware detection & correction** — paste an AI-generated caption;
  the model tags each sentence as hallucinated / non-hallucinated and then
  produces a corrected caption.

This is a **fork of the original `Mobile-O-App`**, stripped down to the
**understanding-only** path. The image-generation half of Mobile-O (DiT
transformer, VAE decoder, diffusion connector) is **not used, not exported, and
not downloaded** — we only ship the vision encoder + LLM. The original
`Mobile-O-App/` folder is untouched.

---

## What differs from the original Mobile-O-App

| Area | Original | This app |
|---|---|---|
| Task | Gen + Understand + Edit + Chat | **VQA + Hallucination only** |
| Exported components | 5 (dit, vae, connector, vision, llm) | **2 (vision, llm)** |
| LLM quantization | 4-bit default | **8-bit default** (medical task, precision-sensitive) |
| Download size | ~3.6 GB | **~1.8 GB** |
| Prompt format | generic "helpful assistant" chat template | **exact finetuning template** (see below) |
| UI | one chat box w/ keyword routing | **two dedicated tabs** (VQA / Hallucination) |
| Model source | `Amshaker/Mobile-O-0.5B-iOS` | **your HF repo** (set in `ModelDownloadManager.swift`) |

### Files changed / added
- `export.py` — rewritten: understanding-only, 8-bit default, defaults to our checkpoint.
- `app/MobileO/App/ContentView.swift` — rewritten: understanding-only loader + 2-tab UI.
- `app/MobileO/Models/Understanding/FastVLM.swift` — `prepare()` gains a **raw-prompt
  passthrough** so the on-device prompt is byte-identical to training.
- `app/MobileO/Services/ModelDownloadManager.swift` — repo + components (vision + llm), sizes.
- `app/MobileO/Views/Download/DownloadPermissionView.swift` — branding + sizes.
- `app/MobileO/Info.plist` — `CFBundleDisplayName = GutVLM`.
- **New:** `app/MobileO/GutVLM/` — `GutVLMModel.swift` (prompts + task runners),
  `VQAView.swift`, `HallucinationView.swift`, `GutVLMShared.swift`.

The original generation/chat Swift files are left in place (they still compile) but
are no longer referenced by the app — dead code you can delete later if you like.

---

## Why the prompt format matters (and how it's handled)

Our model is a small 0.5B network finetuned narrowly, so it is sensitive to the
exact prompt strings it saw in training. Those strings (from the Clariden
`inference.py`) differ from the original app's generic chat template:

- **VQA:** no system prompt; the user turn is closed with `<|im_start|>assistant`
  (not `<|im_end|>`).
- **Hallucination:** a specific medical `SYSTEM_PREFIX`, then a two-turn
  detect → correct conversation with a single `<image>`.

`GutVLMModel` / `GutVLMPrompts` build these strings verbatim, and the patched
`FastVLMProcessor.prepare()` passes them through untouched (only expanding the
single `<image>` placeholder into image tokens). So the device sees the same
prompt bytes as training.

---

## End-to-end setup

### Prerequisites
- **macOS with Xcode 16+** (CoreML export and iOS builds are macOS-only).
- A **physical iPhone 15 or later** (Simulator unsupported — needs CoreML + Metal).
- Python 3.10+ on the Mac for the export step.

### 1. Export the models (on a Mac)

Copy our finetuned checkpoint to the Mac. The checkpoint dir must contain
`model.safetensors`, `config.json`, **and the tokenizer files**
(`tokenizer.json`, `tokenizer_config.json`, `merges.txt`, `vocab.json`,
`special_tokens_map.json`, `added_tokens.json`).

```bash
cd Mobile-O-GutVLM-App
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt      # torch, transformers, timm, coremltools, mlx, ...

# 8-bit (default, recommended)
python export.py /path/to/checkpoints/vlm_gutvlm_hal/epoch_4

# or 4-bit to compare (smaller, may cost accuracy on hallucination task)
python export.py /path/to/checkpoints/vlm_gutvlm_hal/epoch_4 --llm-bits 4
```

Output lands in `exported_models/`:
```
exported_models/
├── vision_encoder.mlpackage
└── llm/
    ├── model.safetensors
    ├── model.safetensors.index.json
    ├── config.json
    └── tokenizer.json, tokenizer_config.json, merges.txt, vocab.json,
        special_tokens_map.json, added_tokens.json
```

> **Sanity check before shipping:** load `exported_models/` in a small MLX script
> and reproduce one `ask()` answer from Clariden. Catch quantization drift here,
> not on the phone.

### 2. Host the weights on HuggingFace

The app downloads weights on first launch (keeps the app binary small). Create a
model repo and upload the exported folder **at the repo root**:

```bash
huggingface-cli login
huggingface-cli repo create Mobile-O-GutVLM-iOS --type model

# from inside exported_models/ so paths land at the repo root
cd exported_models
huggingface-cli upload ankitbelbase034/Mobile-O-GutVLM-iOS . . --repo-type model
```

The repo must end up with this layout (the app fetches these exact paths):
```
<repo root>/
├── vision_encoder.mlpackage/Manifest.json
├── vision_encoder.mlpackage/Data/com.apple.CoreML/model.mlmodel
├── vision_encoder.mlpackage/Data/com.apple.CoreML/weights/weight.bin
└── llm/<all the files above>
```

Then set the repo in `app/MobileO/Services/ModelDownloadManager.swift`:
```swift
private static let repo = "ankitbelbase034/Mobile-O-GutVLM-iOS"   // <-- your repo
```
(It's already set to this — change it if you use a different repo name.)

**Alternative — no HuggingFace / bundle locally:** copy `exported_models/` into
`app/MobileO/Resources/` and load from the bundle instead of downloading. HF is
simpler for keeping the app binary small; bundling is better for a fully offline
demo build. See "Offline bundling" below.

### 3. Build & run

```bash
open app/MobileO.xcodeproj
```
1. Connect your iPhone 15+.
2. In **Signing & Capabilities**, set your Development **Team**.
3. **Cmd + R**.

On first launch the app shows a download gate (~1.8 GB), fetches the two
components from your HF repo, compiles the vision encoder on-device, then opens
the two-tab UI. After that it runs fully offline.

---

### 3b. Or run it natively on a Mac (no iPhone needed)

If you don't have a physical iPhone, `mlx_infer.py` + `mlx_app.py` run the same
Core ML vision encoder + MLX LLM directly on an Apple Silicon Mac via a local
Gradio UI — no Xcode, no PyTorch, no device required.

**Requirements:** an Apple Silicon Mac (M1 or later — Intel Macs can't run MLX).

**Easiest — double-click `GutVLM.command`:**
```bash
git clone https://github.com/ankitbelbase17/Gut_Vlm_finetune.git
```
Then in Finder, open `Gut_Vlm_finetune/Mobile-O/Mobile-O-GutVLM-App/` and
double-click **`GutVLM.command`**. First run sets up a virtual environment and
installs dependencies (a minute or two, one time only); every run after that
just launches the app. It automatically:
- finds a working arm64 Python 3.10+ on your system (MLX needs Apple Silicon —
  the launcher checks for this itself, since a plain `python3` on PATH can
  silently be an Intel/Rosetta build depending on what else you have installed),
- downloads the pre-exported model (~890 MB) from HuggingFace on first run,
- and opens `http://localhost:7860` in your browser once it's ready.

Close the Terminal window it opens (or press Ctrl+C in it) to stop the app.

**From the terminal instead**, if you'd rather manage the environment yourself:
```bash
cd Gut_Vlm_finetune/Mobile-O/Mobile-O-GutVLM-App

# must be an arm64 Python 3.10+ -- check with:
#   python3 -c "import platform,sys; print(platform.machine(), sys.version_info)"
# (Homebrew's `brew install python@3.11` is a reliable source on Apple Silicon)
python3 -m venv .venv-mlx && source .venv-mlx/bin/activate
pip install -r requirements-mlx.txt

python mlx_app.py
```
Either way, with no flags `mlx_app.py` downloads the model from
[`GutVLMmodels/experiments_checkpoints`](https://huggingface.co/GutVLMmodels/experiments_checkpoints/tree/main/gutvlm_epoch4_mlx_coreml)
on first run and caches it locally after that.

**Using your own export or a different HF repo instead** (terminal path only —
edit the `python mlx_app.py` line in `GutVLM.command` if you want the
double-click launcher to use these too):
```bash
# a local export.py output (see §1 above):
python mlx_app.py --exported-dir exported_models --hf-repo ""

# a different HF repo, with vision_encoder.mlpackage/ + llm/ living directly
# at the repo root instead of in a subfolder:
python mlx_app.py --hf-repo <username>/<repo-name> --hf-subfolder ""

# a different HF repo with its own subfolder:
python mlx_app.py --hf-repo <username>/<repo-name> --hf-subfolder <folder>
```

`requirements-mlx.txt` is a much lighter runtime-only dependency set than
`requirements.txt` (no torch/diffusers/timm — those are only needed for the
export step itself, §1 above).

---

## Using the app

**VQA tab** — tap the card to pick an endoscopy image, type a question
(e.g. *"Is there a polyp visible in this image?"*), tap **Ask**.

**Hallucination tab** — pick an image, paste an AI-generated caption, tap
**Detect & Correct**. You get:
- a per-sentence breakdown (✅ non-hallucinated / ⚠️ hallucinated), and
- a corrected caption.

Generation is greedy (temperature 0) to match the deterministic
`do_sample=False` inference on Clariden.

---

## Known differences / things to validate on-device

1. **Image preprocessing.** Training resized images to 1024×1024 directly; the
   iOS `FastVLMProcessor` does `fitIn(shortestEdge)` + `centerCrop`. For
   non-square images this can crop differently. Endoscopy frames are usually
   near-square, so impact is expected to be small — but if VQA answers look off
   vs Clariden, align `preprocessor_config.json` (`shortest_edge` / `crop_size`)
   so the result is a plain 1024² resize.
2. **Quantization.** 8-bit is the default. Spot-check against Clariden outputs; if
   quality holds, `--llm-bits 4` roughly halves the LLM download.
3. **Checkpoint compatibility.** `export.py` loads via the app's
   `MobileOForInferenceLM`. If `from_pretrained` complains about `config.json`,
   confirm our checkpoint's `model_type` matches what that class registers
   (`mobile_o_inference`) — both derive from the same Mobile-O source, so it
   should load unchanged.
4. **Image-token count.** The vision encoder must emit the same number of patches
   the prompt expands `<image>` into (patchSize 64 → 256 tokens at 1024²). Our
   vision tower has the same architecture as base, so this matches; if you change
   the input resolution, keep the two in sync.
5. **`<image>` must be a single token = 151648 (verify this).** On device, the
   image features are spliced at token positions whose id equals
   `image_token_index` (151648). This works only if the tokenizer encodes the
   literal string `<image>` to exactly `[151648]`. Our Clariden training used a
   `-200` sentinel via `tokenizer_image_token()` instead, so **confirm the
   exported tokenizer still maps `<image>` → 151648** — otherwise the merge finds
   zero image positions and the model silently runs text-only (ignores the
   image). Quick check on the Mac after export:
   ```python
   from transformers import AutoTokenizer
   tok = AutoTokenizer.from_pretrained("exported_models/llm")
   print(tok.encode("<image>"))   # must print [151648]
   ```
   If it prints multiple ids, the base Mobile-O tokenizer's `<image>` added-token
   entry is missing from the checkpoint — copy `tokenizer_config.json` /
   `added_tokens.json` / `special_tokens_map.json` from `Mobile-O-0.5B-SFT`.

---

## Offline bundling (optional)

To skip the HF download entirely:
1. Copy `exported_models/*.mlpackage` and `exported_models/llm/` into
   `app/MobileO/Resources/`.
2. Have `MobileOApp` skip the download gate (treat models as ready) and point
   `FastVLM.customModelDirectory` at the bundled `llm/` directory.
   `FastVLM.modelConfiguration` already falls back to the app bundle when
   `customModelDirectory` is nil.

---

## Acknowledgements
Built on top of the original Mobile-O iOS app and
[FastVLM](https://github.com/apple/ml-fastvlm). Understanding-only adaptation for
GI VQA + Gut-VLM hallucination detection.
