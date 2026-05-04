# -*- coding: utf-8 -*-
"""
@author: Rong Qian
@date: 2025/11/21
批量计算电子密度特征
"""

import os
import glob
import multiprocessing as mp

# 你的主函数文件名要一致
from gen_xplor_pdb_npz_multi import generate_xplor_map_from_pdb  

# -----------------------
# 路径配置
# -----------------------

DATA_ROOT = "/public/home/qianrong/projects/MolCRAFT/MolCRAFT/data"
# DATA_ROOT = "/Users/qianrong/vscode/BFN-ED/data"

SRC_DIR = os.path.join(DATA_ROOT, "crossdocked_v1.1_rmsd1.0_pocket8")
# SRC_DIR = os.path.join(DATA_ROOT, "test")


# 多分辨率
RESOLUTIONS = [2.0, 2.5, 3.0, 3.5, 4.0]


def process_one(args):
    pdb_path, out_dir = args
    pdb_name = os.path.basename(pdb_path).replace(".pdb", "")

    # H5文件直接生成在源目录
    h5_out = os.path.join(out_dir, pdb_name + "_multires_pointcloud.h5")
    flag_file = os.path.join(out_dir, pdb_name + "_done.flag")

    # 跳过已经处理的文件
    if os.path.exists(flag_file) and os.path.exists(h5_out):
        print("[SKIP] {}".format(pdb_path))
        return

    try:
        print("[START] {}".format(pdb_path))

        # 直接调用函数，H5文件会生成在PDB文件所在目录
        generate_xplor_map_from_pdb(
            inpdbfile=pdb_path,
            resolution_needed=RESOLUTIONS,
            save_pdb=False,
            keep_xplor=False,
            return_aux=False
        )

        # 写 flag
        with open(flag_file, "w") as f:
            f.write("ok")

        print("[DONE] {}".format(pdb_path))

    except Exception as e:
        print("[ERROR] {}: {}".format(pdb_path, e))


def main():
    # 获取所有子目录
    sub_dirs = next(os.walk(SRC_DIR))[1]
    sub_dirs = sorted(sub_dirs)

    print("[INFO] Found {} subfolders.".format(len(sub_dirs)))

    tasks = []

    for sd in sub_dirs:
        src_sub = os.path.join(SRC_DIR, sd)
        
        # 输出目录就是源目录本身
        pdb_files = glob.glob(os.path.join(src_sub, "*.pdb"))

        for pdb in pdb_files:
            # 每个任务：PDB文件路径 和 输出目录（就是源目录）
            tasks.append((pdb, src_sub))

    print("[INFO] Total PDB files: {}".format(len(tasks)))

    # 使用Slurm分配的核心数
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        cpu = int(os.environ['SLURM_CPUS_PER_TASK'])
    else:
        cpu = min(mp.cpu_count() - 2, 16)  # 回退方案
    
    print("[INFO] Using {} processes".format(cpu))

    # cpu = mp.cpu_count() - 2
    # if cpu < 1:
    #     cpu = 1

    # print("[INFO] Using {} processes".format(cpu))

    pool = mp.Pool(cpu)
    pool.map(process_one, tasks)
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()