"""
Gradio web demo for the Mobile-O hallucination-detection checkpoint.

Run on Clariden (or any machine with the checkpoint and a GPU):
    cd ~/Mobile-O
    pip install gradio
    python app.py --model_path checkpoints/vlm_gutvlm_hal/epoch_4

Then either:
  - Open http://localhost:7860 in a browser on the same machine, OR
  - SSH port-forward: ssh -L 7860:localhost:7860 <clariden-login-node>
    and open http://localhost:7860 locally.

Pass --share to get a public Gradio link (no port forwarding needed):
    python app.py --model_path checkpoints/vlm_gutvlm_hal/epoch_4 --share
"""

import argparse
import sys
import os

MOBILEO_REPO = os.environ.get("MOBILEO_PATH", ".")
if MOBILEO_REPO not in sys.path:
    sys.path.insert(0, MOBILEO_REPO)

import torch
import gradio as gr
from inference import load_model, ask, detect_hallucinations

EXAMPLE_CAPTION = (
    "The endoscopy image shows a large sessile polyp in the sigmoid colon. "
    "There is active bleeding visible from the polyp surface. "
    "The surrounding mucosa appears normal with no signs of inflammation."
)

EXAMPLE_QUESTION = "What abnormalities do you see in this endoscopy image?"


def build_demo(model, tokenizer, image_processor):

    def vqa_fn(image, question, max_tokens):
        if image is None:
            return "Please upload an image."
        if not question.strip():
            return "Please enter a question."
        try:
            return ask(model, tokenizer, image_processor, image, question, int(max_tokens))
        except Exception as e:
            return f"Error: {e}"

    def hallucination_fn(image, caption, max_tokens):
        if image is None:
            return "Please upload an image.", ""
        if not caption.strip():
            return "Please enter a caption to analyse.", ""
        try:
            detection, correction = detect_hallucinations(
                model, tokenizer, image_processor, image, caption, int(max_tokens)
            )
            return detection, correction
        except Exception as e:
            return f"Error: {e}", ""

    with gr.Blocks(title="Mobile-O GI Hallucination Detector") as demo:
        gr.Markdown(
            "# Mobile-O — GI Endoscopy VLM\n"
            "Finetuned on Kvasir-VQA (Step 2) then Gut-VLM hallucination-aware "
            "data (Step 3). Best checkpoint: `epoch_4` (val_loss = 0.1392).\n\n"
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

    return demo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="checkpoints/vlm_gutvlm_hal/epoch_4",
                   help="Path to finetuned checkpoint")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true",
                   help="Create a public Gradio share link")
    args = p.parse_args()

    model, tokenizer, image_processor = load_model(args.model_path, args.device)
    demo = build_demo(model, tokenizer, image_processor)
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
