#!/bin/bash
#SBATCH --job-name=mobile_o_step4
#SBATCH --output=logs/step4_%j.out
#SBATCH --error=logs/step4_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --account=a168
#SBATCH --partition=normal

cd ~/Mobile-O
mkdir -p logs results/step4

python step4_generate_predictions.py \
    --model_path checkpoints/vlm_gutvlm_hal/epoch_4 \
    --test_json  ../Hallucination-Aware-VLM/dataset/Gut-VLM/train_test_split/test.json \
    --images_dir /iopsstor/scratch/cscs/dbartaula/FT/kvasir-v2-flat \
    --output_dir results/step4 \
    --resume
