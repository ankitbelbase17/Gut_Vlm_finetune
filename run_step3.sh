#!/bin/bash
#SBATCH --job-name=mobileo_gut_vlm
#SBATCH -A a168
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --time=01:30:00
#SBATCH --output=/iopsstor/scratch/cscs/dbartaula/FT/logs/%x_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/dbartaula/FT/logs/%x_%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1

WORK_DIR="/iopsstor/scratch/cscs/dbartaula/FT/Mobile-O"
mkdir -p "$WORK_DIR/../logs"
cd "$WORK_DIR"

python step3_finetune_hallucination.py \
    --model_path checkpoints/vlm_kvasir_full_continued/epoch_2 \
    --data data/gut_vlm/train.jsonl \
    --val_data data/gut_vlm/test.jsonl \
    --epochs 6 \
    --output_dir checkpoints/vlm_gutvlm_hal \
    --save_every_steps 50 \
    --eval_every_steps 30 \
    --wandb_project mobile-o-hallucination-finetune \
    --wandb_run_name "clariden-gut-vlm-${SLURM_JOB_ID:-local}"
