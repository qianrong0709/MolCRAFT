#!/bin/bash
#SBATCH -J ed_single
#SBATCH -p 224c2gQ
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
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

# ===== experiment identity =====
exp_name=ed_single
revision=v1
seed=1234

# ===== bool flags =====
debug=false
no_wandb=true
test_only=false
empty_folder=false
resume=false

# ===== hyperparameters =====
sigma1_coord=0.03
beta1=1.5
lr=5e-4
time_emb_dim=1
max_grad_norm=Q
destination_prediction=True
use_discrete_t=True
num_samples=6
sampling_strategy=end_back_pmf

## ===== practical training settings =====
#batch_size=4
#epochs=15

echo "Python path: /public/home/qianrong/.conda/envs/molcraft/bin/python"
/public/home/qianrong/.conda/envs/molcraft/bin/python --version
nvidia-smi

cmd="/public/home/qianrong/.conda/envs/molcraft/bin/python train_bfn.py \
    --config_file configs/default.yaml \
    --exp_name $exp_name \
    --revision $revision \
    --seed $seed \
    --sigma1_coord $sigma1_coord \
    --beta1 $beta1 \
    --lr $lr \
    --time_emb_dim $time_emb_dim \
    --max_grad_norm $max_grad_norm \
    --destination_prediction $destination_prediction \
    --use_discrete_t $use_discrete_t \
    --num_samples $num_samples \
    --sampling_strategy $sampling_strategy"

if [ "$no_wandb" = true ]; then
    cmd="$cmd --no_wandb"
fi

if [ "$debug" = true ]; then
    cmd="$cmd --debug"
fi

if [ "$resume" = true ]; then
    cmd="$cmd --resume"
fi

if [ "$test_only" = true ]; then
    cmd="$cmd --test_only"
fi

if [ "$empty_folder" = true ]; then
    cmd="$cmd --empty_folder"
fi

echo "=================================================="
echo "Final command:"
echo "$cmd"
echo "=================================================="

eval "$cmd"

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="