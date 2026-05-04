import torch
from torch.utils.data import Subset
from core.datasets.pl_pair_dataset import PocketLigandPairDataset, PocketLigandPairDatasetFeaturized, PocketLigandGeneratedPairDataset
from core.datasets.pdbbind import PDBBindDataset


def get_dataset(config, *args, **kwargs):
    name = config.name
    root = config.path
    ligand_atom_mode = config.transform.ligand_atom_mode
    # if name == 'pl':
    #     dataset = PocketLigandPairDataset(root, *args, **kwargs)
    # elif name == 'pl_tr':
    #     dataset = PocketLigandPairDatasetFeaturized(root, ligand_atom_mode=ligand_atom_mode,
    #                                                  *args, **kwargs)
    #     return dataset, {"train": dataset.train_data, "test": dataset.test_data}
    # elif name == 'pl_dcmp':
    #     dataset = PocketLigandGeneratedPairDataset(root, *args, **kwargs)
    #     return dataset, {"train": dataset, "test": dataset}
    # elif name == 'pdbbind':
    #     dataset = PDBBindDataset(root, *args, **kwargs)
    # else:
    #     raise NotImplementedError('Unknown dataset: %s' % name)
    #
    # # print(config)
    #
    # if config.with_split:
    #     split = torch.load(config.split)
    #     subsets = {k: Subset(dataset, indices=v) for k, v in split.items()}
    #     return dataset, subsets
    # else:
    #     return dataset

    # ===== ED config from data section =====
    use_ed = getattr(config, "use_ed", False)
    ed_resolution = getattr(config, "ed_resolution", 3.0)
    ed_k = getattr(config, "ed_k", 32)
    ed_features = tuple(getattr(config, "ed_features", ("rho", "grad_norm", "lap")))
    ed_use_delta_xyz = getattr(config, "ed_use_delta_xyz", True)

    if name == 'pl':
        dataset = PocketLigandPairDataset(
            root,
            *args,
            use_ed=use_ed,
            ed_resolution=ed_resolution,
            ed_k=ed_k,
            ed_features=ed_features,
            ed_use_delta_xyz=ed_use_delta_xyz,
            **kwargs,
        )
    elif name == 'pl_tr':
        dataset = PocketLigandPairDatasetFeaturized(
            root,
            ligand_atom_mode=ligand_atom_mode,
            *args,
            **kwargs,
        )
        return dataset, {"train": dataset.train_data, "test": dataset.test_data}
    elif name == 'pl_dcmp':
        dataset = PocketLigandGeneratedPairDataset(root, *args, **kwargs)
        return dataset, {"train": dataset, "test": dataset}
    elif name == 'pdbbind':
        dataset = PDBBindDataset(root, *args, **kwargs)
    else:
        raise NotImplementedError('Unknown dataset: %s' % name)

    if config.with_split:
        split = torch.load(config.split)
        subsets = {k: Subset(dataset, indices=v) for k, v in split.items()}
        return dataset, subsets
    else:
        return dataset
