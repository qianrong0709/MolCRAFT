#!/bin/bash
#SBATCH --job-name=extract_pocket_job
#SBATCH --output=logs/extract_pocket_%j.out
#SBATCH --error=logs/extract_pocket_%j.err
#SBATCH --partition=112c3gQ
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=168:00:00     # 168小时 = 7天


echo "=== Job started at $(date) ==="
echo "Node: $SLURM_NODELIST"

# ==========================
# 1. 最通用：自动定位 conda
# ==========================
eval "$($(which conda) shell.bash hook)"
conda activate molcraft    # ⚠️ 改成你的环境名字

# ==========================
# 2. 切换到 MolCRAFT 项目目录
# ==========================
cd /public/home/qianrong/projects/MolCRAFT/MolCRAFT
echo "Working directory: $(pwd)"

# ==========================
# 3. 开始运行 pocket 提取
# ==========================
python -m core.datasets.extract_pockets \
    --source data/crossdocked_v1.1_rmsd1.0 \
    --dest   data/crossdocked_v1.1_rmsd1.0_pocket8 \
    --radius 8 \
    --num_workers 16

echo "=== Job finished at $(date) ==="