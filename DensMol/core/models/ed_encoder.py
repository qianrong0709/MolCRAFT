# -*- coding: utf-8 -*-
"""
@author: Rong Qian
@date: 2026/03/11

ED local point cloud encoder
输入:
    local_ed: [N_atom, K, C]
输出:
    ed_emb:   [N_atom, ed_dim]
"""

import torch
import torch.nn as nn


class EDPointNetEncoder(nn.Module):
    """
    最小版 PointNet-style encoder:
        per-point MLP + max pooling over K

    输入:
        local_ed: [N_atom, K, C]
    输出:
        ed_emb:   [N_atom, ed_dim]
    """
    def __init__(self, input_dim=6, hidden_dim=32, ed_dim=64, pool="max"):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ed_dim = ed_dim
        self.pool = pool

        self.point_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ed_dim),
            nn.ReLU(),
        )

    def forward(self, local_ed):
        """
        local_ed: [N_atom, K, C]
        return:
            ed_emb: [N_atom, ed_dim]
        """
        if local_ed.dim() != 3:
            raise ValueError(f"local_ed must be 3D [N_atom, K, C], got shape {tuple(local_ed.shape)}")

        n_atom, k, c = local_ed.shape
        if c != self.input_dim:
            raise ValueError(
                f"local_ed last dim mismatch: expected {self.input_dim}, got {c}"
            )

        # [N_atom, K, C] -> [N_atom*K, C]
        x = local_ed.reshape(n_atom * k, c)

        # per-point MLP
        x = self.point_mlp(x)   # [N_atom*K, ed_dim]

        # [N_atom*K, ed_dim] -> [N_atom, K, ed_dim]
        x = x.reshape(n_atom, k, self.ed_dim)

        # pool over K
        if self.pool == "max":
            ed_emb = x.max(dim=1).values   # [N_atom, ed_dim]
        elif self.pool == "mean":
            ed_emb = x.mean(dim=1)         # [N_atom, ed_dim]
        else:
            raise ValueError(f"Unsupported pool type: {self.pool}")

        return ed_emb


def _test_ed_encoder():
    """
    用随机张量做一个最小 sanity check
    """
    local_ed = torch.randn(246, 32, 6)
    model = EDPointNetEncoder(input_dim=6, hidden_dim=32, ed_dim=64, pool="max")

    ed_emb = model(local_ed)

    # print("=" * 80)
    # print("[TEST] local_ed.shape =", tuple(local_ed.shape))
    # print("[TEST] ed_emb.shape   =", tuple(ed_emb.shape))
    # print("[TEST] has_nan        =", torch.isnan(ed_emb).any().item())
    # print("[TEST] has_inf        =", torch.isinf(ed_emb).any().item())
    # print("[TEST] dtype          =", ed_emb.dtype)
    # print("[TEST] min/max        =", float(ed_emb.min()), float(ed_emb.max()))
    # print("=" * 80)


if __name__ == "__main__":
    _test_ed_encoder()