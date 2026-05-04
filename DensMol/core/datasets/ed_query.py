# -*- coding: utf-8 -*-
"""
@author: Rong Qian
@date: 2026/03/11

数据处理：ED查询
"""

import os
import h5py
import torch
import numpy as np
import lmdb
import pickle
from core.datasets.pl_data import ProteinLigandData


def load_ed_from_h5(h5_path, resolution=3.0, features=("rho", "grad_norm", "lap")):
    """
    从一个 h5 文件中读取指定分辨率下的 ED 点云坐标和特征。

    返回:
        ed_xyz:   torch.FloatTensor, [N_e, 3]
        ed_feats: torch.FloatTensor, [N_e, F]
    """
    res = str(resolution)

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"h5 file not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        group_path = f"/data/{res}"
        if group_path not in f:
            raise KeyError(f"{group_path} not found in {h5_path}")

        g = f[group_path]

        ed_xyz = torch.from_numpy(g["xyz"][()].astype(np.float32))   # [N_e, 3]

        feat_list = []
        for feat_name in features:
            if feat_name not in g:
                raise KeyError(f"feature '{feat_name}' not found in {group_path}")
            arr = g[feat_name][()].astype(np.float32)   # [N_e]
            feat_list.append(arr)

        ed_feats = np.stack(feat_list, axis=-1)         # [N_e, F]
        ed_feats = torch.from_numpy(ed_feats)

    return ed_xyz, ed_feats


def query_local_ed(
    protein_pos,
    h5_path,
    resolution=3.0,
    k=32,
    features=("rho", "grad_norm", "lap"),
    use_delta_xyz=True,
):
    """
    对每个 pocket 原子，在 ED 点云中找最近的 k 个点，返回局部 ED 特征。

    输入:
        protein_pos: torch.Tensor [N_p, 3]
        h5_path: str
        resolution: float
        k: int
        features: tuple/list, e.g. ("rho", "grad_norm", "lap")
        use_delta_xyz: bool

    返回:
        local_ed: torch.Tensor [N_p, k, C]
            若 use_delta_xyz=True 且 features=3 个，则 C=6
    """
    if not torch.is_tensor(protein_pos):
        protein_pos = torch.tensor(protein_pos, dtype=torch.float32)
    protein_pos = protein_pos.float()  # [N_p, 3]

    ed_xyz, ed_feats = load_ed_from_h5(
        h5_path=h5_path,
        resolution=resolution,
        features=features,
    )

    # 保证都在 CPU 上，先写最稳版本
    protein_pos = protein_pos.cpu()
    ed_xyz = ed_xyz.cpu()
    ed_feats = ed_feats.cpu()

    # pairwise distance: [N_p, N_e]
    dist = torch.cdist(protein_pos, ed_xyz)

    # 取最近 k 个点
    # knn_idx: [N_p, k]
    # knn_dist, knn_idx = torch.topk(dist, k=k, dim=1, largest=False)
    num_ed = ed_xyz.size(0)
    k_eff = min(k, num_ed)

    knn_dist, knn_idx = torch.topk(dist, k=k_eff, dim=1, largest=False)

    # 取对应的 ed 点坐标 / ed 特征
    # ed_xyz[knn_idx] -> [N_p, k, 3]
    local_xyz = ed_xyz[knn_idx]
    local_feats = ed_feats[knn_idx]   # [N_p, k, F]

    if k_eff < k:
        pad_size = k - k_eff
        pad_xyz = local_xyz[:, -1:, :].expand(-1, pad_size, -1)
        pad_feats = local_feats[:, -1:, :].expand(-1, pad_size, -1)
        local_xyz = torch.cat([local_xyz, pad_xyz], dim=1)
        local_feats = torch.cat([local_feats, pad_feats], dim=1)

    if use_delta_xyz:
        # protein_pos[:, None, :] -> [N_p, 1, 3]
        delta_xyz = local_xyz - protein_pos[:, None, :]   # [N_p, k, 3]
        local_ed = torch.cat([delta_xyz, local_feats], dim=-1)
    else:
        local_ed = local_feats

    return local_ed.float()


def test_query_one_sample(sample, h5_path, resolution=3.0, k=32):
    """
    用一条 lmdb 样本做简单测试。
    sample 需要至少包含:
        sample["protein_pos"]
    """
    protein_pos = sample["protein_pos"] if isinstance(sample, dict) else sample.protein_pos

    local_ed = query_local_ed(
        protein_pos=protein_pos,
        h5_path=h5_path,
        resolution=resolution,
        k=k,
        features=("rho", "grad_norm", "lap"),
        use_delta_xyz=True,
    )

    print("protein_pos.shape =", tuple(protein_pos.shape))
    print("h5_path           =", h5_path)
    print("local_ed.shape    =", tuple(local_ed.shape))
    print("has_nan           =", torch.isnan(local_ed).any().item())
    print("first atom / first 3 neighbors =")
    print(local_ed[0, :3])

    return local_ed

############################## test code ##############################

def _sample_to_h5_path(sample, raw_root: str, pocket_suffix: str = "pocket8"):
    """
    根据 lmdb 读出来的一条 sample，推导对应 h5 路径
    """
    if isinstance(sample, dict):
        ligand_filename = sample["ligand_filename"]
    else:
        ligand_filename = sample.ligand_filename

    ligand_path = os.path.join(raw_root, ligand_filename)
    base = ligand_path[:-4]   # 去掉 .sdf
    h5_path = f"{base}_{pocket_suffix}_multires_pointcloud.h5"
    return h5_path


def _load_one_lmdb_sample(lmdb_path: str, idx: int = 0):
    """
    从 lmdb 里读取一条样本
    """
    env = lmdb.open(
        lmdb_path,
        readonly=True,
        lock=False,
        readahead=False,
        subdir=False,
    )

    with env.begin() as txn:
        cur = txn.cursor()
        if not cur.first():
            raise RuntimeError(f"Empty lmdb: {lmdb_path}")

        for _ in range(idx):
            if not cur.next():
                raise IndexError(f"idx={idx} out of range")

        _, v = cur.item()
        data = pickle.loads(v)

    env.close()

    # lmdb 里存的是 dict，这里转回 ProteinLigandData
    data = ProteinLigandData(**data)
    data.id = idx
    return data


def _test_query_one_sample():   # 内部用，不打算当成公开接口给别人调用
    """
    Day1 单样本测试入口
    """
    lmdb_path = "./data/crossdocked_v1.1_rmsd1.0_pocket8_processed_final.lmdb"
    raw_root = "./data/crossdocked_v1.1_rmsd1.0_pocket8"

    sample = _load_one_lmdb_sample(lmdb_path, idx=0)
    h5_path = _sample_to_h5_path(sample, raw_root=raw_root, pocket_suffix="pocket8")

    print("=" * 80)
    print("[TEST] sample id        :", sample.id)
    print("[TEST] ligand_filename  :", sample.ligand_filename)
    print("[TEST] protein_filename :", sample.protein_filename)
    print("[TEST] h5_path          :", h5_path)
    print("[TEST] h5 exists        :", os.path.exists(h5_path))
    print("[TEST] protein_pos.shape:", tuple(sample.protein_pos.shape))

    local_ed = test_query_one_sample(
        sample=sample,
        h5_path=h5_path,
        resolution=3.0,
        k=32,
    )

    print("-" * 80)
    print("[TEST] has_inf          :", torch.isinf(local_ed).any().item())
    print("[TEST] dtype            :", local_ed.dtype)
    print("[TEST] local_ed min/max :", float(local_ed.min()), float(local_ed.max()))
    print("=" * 80)

    return local_ed


if __name__ == "__main__":
    _test_query_one_sample()