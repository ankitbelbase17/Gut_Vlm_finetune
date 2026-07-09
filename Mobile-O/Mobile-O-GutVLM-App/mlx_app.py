"""
Gradio web demo for the finetuned GutVLM checkpoint, running natively on macOS via
Core ML (vision) + MLX (LLM) -- no iPhone, no Xcode, no PyTorch at inference time.

Run after exporting with export.py:
    conda activate gutvlm-mlx
    python mlx_app.py --exported-dir exported_models

Or, if you don't have a local export, just run it with no extra setup --
by default this downloads the pre-exported model from HuggingFace Hub
(GutVLMmodels/experiments_checkpoints, folder gutvlm_epoch4_mlx_coreml/)
on first run:
    python mlx_app.py

To point at a different HF repo/folder, or skip HF entirely and use your own
local export.py output:
    python mlx_app.py --hf-repo <username>/<repo-name> --hf-subfolder <folder>
    python mlx_app.py --exported-dir exported_models --hf-repo ""

Then open http://localhost:7860 in a browser on this Mac.
"""

import argparse
import json
from pathlib import Path

import gradio as gr
from mlx_infer import load_state, ask, detect_hallucinations

# Expected relative paths inside an exported_models/ directory.
_VISION_MANIFEST = "vision_encoder.mlpackage/Manifest.json"
_LLM_WEIGHTS = "llm/model.safetensors"

# Where the pre-exported model lives by default, so a fresh clone can just
# run `python mlx_app.py` with no other setup.
_DEFAULT_HF_REPO = "GutVLMmodels/experiments_checkpoints"
_DEFAULT_HF_SUBFOLDER = "gutvlm_epoch4_mlx_coreml"

EXAMPLE_CAPTION = (
    "The endoscopy image shows a large sessile polyp in the sigmoid colon. "
    "There is active bleeding visible from the polyp surface. "
    "The surrounding mucosa appears normal with no signs of inflammation."
)

EXAMPLE_QUESTION = "What abnormalities do you see in this endoscopy image?"

# demo_data/vqa/ ships in the repo two levels up from this file
# (Mobile-O/Mobile-O-GutVLM-App/mlx_app.py -> repo root -> demo_data/vqa/).
DEMO_DATA_DIR = Path(__file__).resolve().parents[2] / "demo_data" / "vqa"


def _load_vqa_examples():
    """Load demo_data/vqa/questions.json, resolving each entry's image path
    against DEMO_DATA_DIR and skipping any whose image file isn't present."""
    questions_path = DEMO_DATA_DIR / "questions.json"
    if not questions_path.exists():
        return []
    records = json.loads(questions_path.read_text())
    examples = []
    for r in records:
        image_path = DEMO_DATA_DIR / "images" / Path(r["image"].replace("\\", "/")).name
        if image_path.exists():
            examples.append([str(image_path), r["question"]])
    return examples


def _demo_image_paths():
    """All demo images, for use as image-only examples (e.g. in the
    Hallucination Detection tab, which has no matching demo captions)."""
    images_dir = DEMO_DATA_DIR / "images"
    if not images_dir.exists():
        return []
    return sorted(str(p) for p in images_dir.glob("*.jpg"))


def ensure_exported_models(exported_dir: Path, hf_repo: str = None, hf_subfolder: str = None) -> Path:
    """Return a local directory containing vision_encoder.mlpackage/ and llm/,
    downloading it from a HuggingFace Hub repo first if not already present.

    hf_subfolder matters because snapshot_download() preserves the repo's own
    directory structure -- if the files live under e.g. gutvlm_epoch4_mlx_coreml/
    in the repo (rather than at the repo root), the downloaded copy lands at
    exported_dir/gutvlm_epoch4_mlx_coreml/... too, so the *returned* path is
    exported_dir/hf_subfolder, not exported_dir itself.

    Mirrors the download-on-first-launch pattern the iOS app uses in
    ModelDownloadManager.swift, but for a plain local directory instead of
    the app's Application Support folder.
    """
    local_root = (exported_dir / hf_subfolder) if hf_subfolder else exported_dir

    vision_ok = (local_root / _VISION_MANIFEST).exists()
    llm_ok = (local_root / _LLM_WEIGHTS).exists()
    if vision_ok and llm_ok:
        return local_root

    if not hf_repo:
        raise FileNotFoundError(
            f"No exported model found at '{local_root}' (missing {_VISION_MANIFEST} "
            f"and/or {_LLM_WEIGHTS}).\n"
            "Either run export.py first, or pass --hf-repo <username>/<repo-name> "
            "(and --hf-subfolder if applicable) to download a pre-exported copy "
            "from HuggingFace Hub."
        )

    prefix = f"{hf_subfolder}/" if hf_subfolder else ""
    print(f"[mlx_app] '{local_root}' not found locally -- downloading from "
          f"HuggingFace Hub repo '{hf_repo}'{f' (folder {hf_subfolder}/)' if hf_subfolder else ''} ...")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=hf_repo,
        local_dir=str(exported_dir),
        allow_patterns=[f"{prefix}vision_encoder.mlpackage/*", f"{prefix}llm/*"],
    )
    print(f"[mlx_app] Downloaded to {local_root}")
    return local_root


def build_demo(state):

    def vqa_fn(image, question, max_tokens):
        if image is None:
            return "Please upload an image."
        if not question.strip():
            return "Please enter a question."
        try:
            return ask(state, image, question, int(max_tokens))
        except Exception as e:
            return f"Error: {e}"

    def hallucination_fn(image, caption, max_tokens):
        if image is None:
            return "Please upload an image.", ""
        if not caption.strip():
            return "Please enter a caption to analyse.", ""
        try:
            detection, correction = detect_hallucinations(state, image, caption, int(max_tokens))
            return detection, correction
        except Exception as e:
            return f"Error: {e}", ""

    with gr.Blocks(title="Mobile-O GI Hallucination Detector (MLX, on-device)") as demo:
        gr.Markdown(
            "# Mobile-O — GI Endoscopy VLM (native macOS: Core ML + MLX)\n"
            "Finetuned on Kvasir-VQA (Step 2) then Gut-VLM hallucination-aware "
            "data (Step 3). Running the Core ML vision encoder + MLX Qwen2 LLM "
            "exported by `export.py`, entirely on this Mac.\n\n"
            "**Two modes below** — VQA for general questions, Hallucination "
            "Detection to tag and correct hallucinated sentences in an AI-generated caption."
        )

        with gr.Tab("VQA Mode"):
            gr.Markdown("Upload an endoscopy image and ask any question about it.")
            with gr.Row():
                with gr.Column(scale=1):
                    vqa_image = gr.Image(type="pil", label="Endoscopy Image")
                    vqa_question = gr.Textbox(
                        label="Question",
                        value=EXAMPLE_QUESTION,
                        lines=2,
                    )
                    vqa_tokens = gr.Slider(64, 512, value=256, step=32, label="Max new tokens")
                    vqa_btn = gr.Button("Ask", variant="primary")
                with gr.Column(scale=1):
                    vqa_answer = gr.Textbox(label="Answer", lines=8, interactive=False)
            vqa_btn.click(vqa_fn, [vqa_image, vqa_question, vqa_tokens], vqa_answer)

            vqa_examples = _load_vqa_examples()
            if vqa_examples:
                gr.Examples(
                    examples=vqa_examples,
                    inputs=[vqa_image, vqa_question],
                    label="Example images (from demo_data/vqa/)",
                )

        with gr.Tab("Hallucination Detection Mode"):
            gr.Markdown(
                "Upload an endoscopy image and paste an AI-generated caption. "
                "The model will:\n"
                "1. Tag each sentence as `<hallucinated>` or `<non-hallucinated>`\n"
                "2. Produce a corrected version of the caption"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    hal_image = gr.Image(type="pil", label="Endoscopy Image")
                    hal_caption = gr.Textbox(
                        label="AI-generated caption to check",
                        value=EXAMPLE_CAPTION,
                        lines=5,
                    )
                    hal_tokens = gr.Slider(64, 512, value=384, step=32, label="Max new tokens")
                    hal_btn = gr.Button("Detect & Correct", variant="primary")
                with gr.Column(scale=1):
                    hal_detection = gr.Textbox(
                        label="Hallucination Detection (per sentence)",
                        lines=6,
                        interactive=False,
                    )
                    hal_correction = gr.Textbox(
                        label="Corrected Caption",
                        lines=6,
                        interactive=False,
                    )
            hal_btn.click(
                hallucination_fn,
                [hal_image, hal_caption, hal_tokens],
                [hal_detection, hal_correction],
            )

            demo_images = _demo_image_paths()
            if demo_images:
                gr.Examples(
                    examples=[[p] for p in demo_images],
                    inputs=[hal_image],
                    label="Example images (from demo_data/vqa/) — pair with the caption above",
                )

    return demo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exported-dir", default="exported_models",
                    help="Local directory for the exported model (downloaded here if missing "
                         "and --hf-repo is set, or produced here by export.py)")
    p.add_argument("--hf-repo", default=_DEFAULT_HF_REPO,
                    help=f"HuggingFace Hub model repo (username/repo-name) to download from if "
                         f"--exported-dir doesn't exist locally. Default: {_DEFAULT_HF_REPO}. "
                         "Pass an empty string to disable auto-download and require a local export.")
    p.add_argument("--hf-subfolder", default=_DEFAULT_HF_SUBFOLDER,
                    help="Subfolder within --hf-repo containing vision_encoder.mlpackage/ and "
                         f"llm/ (default: {_DEFAULT_HF_SUBFOLDER}). Pass an empty string if your "
                         "repo has them at the repo root instead.")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()

    exported_dir = ensure_exported_models(
        Path(args.exported_dir), args.hf_repo or None, args.hf_subfolder or None
    )

    print(f"[mlx_app] Loading exported model from {exported_dir} ...")
    state = load_state(exported_dir)
    print("[mlx_app] Model loaded.")

    demo = build_demo(state)
    # demo_data/vqa/images lives outside this app's cwd, so Gradio needs it
    # explicitly allow-listed to serve the example images.
    demo.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        allowed_paths=[str(DEMO_DATA_DIR / "images")],
    )


if __name__ == "__main__":
    main()
