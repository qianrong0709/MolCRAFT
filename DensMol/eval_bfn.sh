#!/bin/bash
#SBATCH -J ed_eval
#SBATCH -p 224c2gQ
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "=================================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "=================================================="

cd /public/home/qianrong/projects/MolCRAFT/MolCRAFT || exit 1
mkdir -p logs

PYTHON_BIN=/public/home/qianrong/.conda/envs/molcraft/bin/python

echo "Python path: $PYTHON_BIN"
$PYTHON_BIN --version
nvidia-smi

echo "=================================================="
echo "Running evaluation..."
echo "=================================================="

$PYTHON_BIN train_bfn.py \
    --config_file configs/default.yaml \
    --test_only \
    --num_samples 10 \
    --sample_steps 100 \
    --no_wandb \
    --ckpt_path /public/home/qianrong/projects/MolCRAFT/MolCRAFT/logs/qianrong_ed_bfn/ed_single/v1_50epoch/checkpoints/last.ckpt

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="