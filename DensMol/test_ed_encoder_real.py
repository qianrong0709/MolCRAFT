import torch
from core.datasets.pl_pair_dataset import PocketLigandPairDataset
from core.models.ed_encoder import EDPointNetEncoder

dataset = PocketLigandPairDataset(
    './data/crossdocked_v1.1_rmsd1.0_pocket8',
    use_ed=True,
    ed_resolution=3.0,
    ed_k=32,
    ed_features=("rho", "grad_norm", "lap"),
    ed_use_delta_xyz=True,
)

data = dataset[0]
local_ed = data.local_ed   # [N_atom, 32, 6]

model = EDPointNetEncoder(input_dim=6, hidden_dim=32, ed_dim=64, pool="max")
ed_emb = model(local_ed)

print("local_ed.shape =", tuple(local_ed.shape))
print("ed_emb.shape   =", tuple(ed_emb.shape))
print("has_nan        =", torch.isnan(ed_emb).any().item())
print("has_inf        =", torch.isinf(ed_emb).any().item())
print("min/max        =", float(ed_emb.min()), float(ed_emb.max()))