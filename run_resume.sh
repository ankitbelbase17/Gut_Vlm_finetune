#!/bin/bash
#SBATCH --job-name=mobileo_vlm_resume
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

python ../step2_finetune_refined.py \
    --data data/train.jsonl \
    --epochs 3 \
    --output_dir checkpoints/vlm_kvasir_full_continued \
    --resume_from checkpoints/vlm_kvasir_full_continued/latest \
    --save_every_steps 200 \
    --eval_every_steps 500 \
    --wandb_project mobile-o-vlm-finetune \
    --wandb_run_name "clariden-resume-${SLURM_JOB_ID:-local}"
