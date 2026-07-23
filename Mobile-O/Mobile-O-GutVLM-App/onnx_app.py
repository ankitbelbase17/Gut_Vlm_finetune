"""
Gradio web demo for the finetuned GutVLM checkpoint, running via ONNX Runtime on
the CPU of ANY laptop (Windows / Linux / Intel-Mac / Apple-Mac) -- no CoreML, no
MLX, no GPU, no PyTorch at inference time.

Cross-platform sibling of mlx_app.py (which is Apple-Silicon-only). Same two modes
(VQA + hallucination detection/correction), same finetuned model, sourced from the
three ONNX graphs produced by onnx_export.py.

Run after exporting with onnx_export.py:
    pip install -r requirements-onnx.txt gradio
    python onnx_app.py --onnx-dir onnx_models

Or, with no local export, just run it -- by default it downloads the pre-exported
ONNX model from HuggingFace Hub (GutVLMmodels/experiments_checkpoints, folder
gutvlm_epoch4_onnx/) on first run:
    python onnx_app.py

Point at a different repo/folder, or skip HF and use your own local export:
    python onnx_app.py --hf-repo <username>/<repo-name> --hf-subfolder <folder>
    python onnx_app.py --onnx-dir onnx_models --hf-repo ""

Then open http://localhost:7860 in a browser.
"""

import argparse
import json
from pathlib import Path

import gradio as gr
from onnx_infer import load_state, ask, detect_hallucinations

# Files that must be present in an onnx_models/ directory for it to be usable.
_REQUIRED = ["vision_encoder.onnx", "embed_tokens.onnx", "decoder.onnx",
             "runtime_config.json", "tokenizer.json"]

# Where the pre-exported ONNX model lives by default, so a fresh clone can just
# run `python onnx_app.py` with no other setup.
_DEFAULT_HF_REPO = "GutVLMmodels/experiments_checkpoints"
_DEFAULT_HF_SUBFOLDER = "gutvlm_epoch4_onnx"

EXAMPLE_CAPTION = (
    "The endoscopy image shows a large sessile polyp in the sigmoid colon. "
    "There is active bleeding visible from the polyp surface. "
    "The surrounding mucosa appears normal with no signs of inflammation."
)
EXAMPLE_QUESTION = "What abnormalities do you see in this endoscopy image?"

DEMO_DATA_DIR = Path(__file__).resolve().parents[2] / "demo_data" / "vqa"


def _load_vqa_examples():
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
    images_dir = DEMO_DATA_DIR / "images"
    if not images_dir.exists():
        return []
    return sorted(str(p) for p in images_dir.glob("*.jpg"))


def ensure_onnx_models(onnx_dir: Path, hf_repo: str = None, hf_subfolder: str = None) -> Path:
    """Return a local directory containing the three ONNX graphs + tokenizer,
    downloading it from HuggingFace Hub first if not already present.

    Like mlx_app.ensure_exported_models(): snapshot_download() preserves the
    repo's directory structure, so if the files live under gutvlm_epoch4_onnx/
    in the repo, the returned path is onnx_dir/hf_subfolder, not onnx_dir itself.
    """
    local_root = (onnx_dir / hf_subfolder) if hf_subfolder else onnx_dir

    if all((local_root / f).exists() for f in _REQUIRED):
        return local_root

    if not hf_repo:
        missing = [f for f in _REQUIRED if not (local_root / f).exists()]
        raise FileNotFoundError(
            f"No exported ONNX model found at '{local_root}' (missing: {missing}).\n"
            "Either run onnx_export.py first, or pass --hf-repo <username>/<repo-name> "
            "(and --hf-subfolder if applicable) to download a pre-exported copy."
        )

    prefix = f"{hf_subfolder}/" if hf_subfolder else ""
    print(f"[onnx_app] '{local_root}' not found locally -- downloading from "
          f"HuggingFace Hub repo '{hf_repo}'"
          f"{f' (folder {hf_subfolder}/)' if hf_subfolder else ''} ...")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=hf_repo,
        local_dir=str(onnx_dir),
        allow_patterns=[f"{prefix}*"],
    )
    print(f"[onnx_app] Downloaded to {local_root}")
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
            return detect_hallucinations(state, image, caption, int(max_tokens))
        except Exception as e:
            return f"Error: {e}", ""

    with gr.Blocks(title="Mobile-O GI Hallucination Detector (ONNX, cross-platform)") as demo:
        gr.Markdown(
            "# Mobile-O — GI Endoscopy VLM (ONNX Runtime, runs on any laptop)\n"
            "Finetuned on Kvasir-VQA (Step 2) then Gut-VLM hallucination-aware "
            "data (Step 3). Running the ONNX vision encoder + Qwen2 LLM exported by "
            "`onnx_export.py`, entirely on this machine's CPU — no GPU, no Apple "
            "Silicon required.\n\n"
            "**Two modes below** — VQA for general questions, Hallucination "
            "Detection to tag and correct hallucinated sentences in an AI-generated caption."
        )

        with gr.Tab("VQA Mode"):
            gr.Markdown("Upload an endoscopy image and ask any question about it.")
            with gr.Row():
                with gr.Column(scale=1):
                    vqa_image = gr.Image(type="pil", label="Endoscopy Image")
                    vqa_question = gr.Textbox(label="Question", value=EXAMPLE_QUESTION, lines=2)
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
                        label="AI-generated caption to check", value=EXAMPLE_CAPTION, lines=5,
                    )
                    hal_tokens = gr.Slider(64, 512, value=384, step=32, label="Max new tokens")
                    hal_btn = gr.Button("Detect & Correct", variant="primary")
                with gr.Column(scale=1):
                    hal_detection = gr.Textbox(
                        label="Hallucination Detection (per sentence)", lines=6, interactive=False,
                    )
                    hal_correction = gr.Textbox(
                        label="Corrected Caption", lines=6, interactive=False,
                    )
            hal_btn.click(
                hallucination_fn, [hal_image, hal_caption, hal_tokens],
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
    p.add_argument("--onnx-dir", default="onnx_models",
                   help="Local directory for the exported ONNX model (downloaded here if "
                        "missing and --hf-repo is set, or produced here by onnx_export.py)")
    p.add_argument("--hf-repo", default=_DEFAULT_HF_REPO,
                   help=f"HuggingFace Hub model repo to download from if --onnx-dir doesn't "
                        f"exist locally. Default: {_DEFAULT_HF_REPO}. Empty string disables it.")
    p.add_argument("--hf-subfolder", default=_DEFAULT_HF_SUBFOLDER,
                   help=f"Subfolder within --hf-repo containing the ONNX files "
                        f"(default: {_DEFAULT_HF_SUBFOLDER}). Empty string if at repo root.")
    p.add_argument("--threads", type=int, default=0, help="ONNX Runtime intra-op threads (0=auto)")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    args = p.parse_args()

    onnx_dir = ensure_onnx_models(
        Path(args.onnx_dir), args.hf_repo or None, args.hf_subfolder or None
    )

    print(f"[onnx_app] Loading ONNX model from {onnx_dir} ...")
    state = load_state(onnx_dir, threads=args.threads)
    print("[onnx_app] Model loaded.")

    demo = build_demo(state)
    demo.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        allowed_paths=[str(DEMO_DATA_DIR / "images")],
        share=args.share,
    )


if __name__ == "__main__":
    main()
