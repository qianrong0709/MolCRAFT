#!/bin/bash
#SBATCH --job-name=map_gen_job
#SBATCH --output=logs/map_gen_%j.out
#SBATCH --error=logs/map_gen_%j.err
#SBATCH --partition=224c2gQ
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=336:00:00     # 168小时 = 7天, 可根据需要调整14天


source ~/.bashrc
conda activate molcraft

echo "[INFO] Start map generation at $(date)"
echo "[INFO] CPU cores available: $(nproc)"

phenix.python /public/home/qianrong/projects/MolCRAFT/MolCRAFT/core/datasets/batch_map.py

echo "[INFO] Finished at $(date)"
