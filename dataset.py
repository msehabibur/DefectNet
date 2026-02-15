"""
dataset.py
Dataset & DataLoader for the merged_data.csv force-field format.

Required CSV columns
--------------------
Structure      :  JSON-serialised pymatgen Structure
Energy         :  float  [eV]        – total DFT energy
Forces         :  str    [eV/Å]      – Python list-of-lists  [[fx,fy,fz], ...]
Stress         :  str    [kBar]      – 3×3 Python list-of-lists (VASP convention)
Charge         :  float              – system charge (e.g. -2, -1, 0, +1, +2)
LevelOfTheory  :  str                – DFT functional: "HSE", "PBE", "PBEsol", …

Optional CSV columns
--------------------
Directory  :  str                 – identifier / path
Frequency  :  int                 – sampling weight (unused here)
CFE        :  float               – cohesive / formation energy (optional)
Tag        :  str                 – e.g. "bulk", "defect", ... (not used by model)
"""

import ast
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from pymatgen.core import Structure
from graph_utils import build_crystal_graph


# kBar  →  eV/ų   (1 eV/ų = 160.2176634 GPa = 1602.1766 kBar)
KBAR_TO_EV_PER_A3 = 1.0 / 1602.1766208

# Global feature encoding maps
THEORY_MAP = {"hse": 0, "pbe": 1, "pbesol": 2, "scan": 3, "lda": 4}


class ForceFieldDataset(Dataset):
    """
    Reads ``merged_data.csv`` and builds crystal graphs on the fly.

    Parameters
    ----------
    csv_path       : path to the CSV file
    cutoff         : neighbour cutoff in Angstrom
    max_neighbors  : max neighbours per atom
    cache_dir      : if given, cache each processed graph as a .pt file
    stress_unit    : "kbar" or "ev_per_a3" – unit of stress in the CSV
    """

    def __init__(
        self,
        csv_path: str,
        cutoff: float = 5.0,
        max_neighbors: int = 12,
        cache_dir=None,
        stress_unit: str = "kbar",
    ):
        self.df = pd.read_csv(csv_path)
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.cache_dir = cache_dir
        self.stress_unit = stress_unit.lower()

        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # --- check disk cache ---
        if self.cache_dir is not None:
            cache_path = os.path.join(self.cache_dir, f"{idx}.pt")
            if os.path.exists(cache_path):
                return torch.load(cache_path, weights_only=False)

        row = self.df.iloc[idx]

        # --- parse pymatgen Structure ---
        structure = Structure.from_dict(json.loads(row["Structure"]))

        # --- build crystal graph ---
        graph = build_crystal_graph(
            structure, cutoff=self.cutoff, max_neighbors=self.max_neighbors,
        )

        # --- parse targets ---
        energy = float(row["Energy"])
        forces = np.array(ast.literal_eval(row["Forces"]), dtype=np.float64)
        stress = np.array(ast.literal_eval(row["Stress"]), dtype=np.float64)

        # convert stress to eV/ų if needed
        if self.stress_unit == "kbar":
            stress = stress * KBAR_TO_EV_PER_A3

        # --- parse global features: charge, level of theory ---
        charge = float(row["Charge"])

        theory_str = str(row["LevelOfTheory"]).strip().lower()
        theory_id = float(THEORY_MAP.get(theory_str, 0))

        data = {
            "atom_types":      torch.tensor(graph["atom_types"], dtype=torch.long),
            "pos":             torch.tensor(graph["positions"],  dtype=torch.float32),
            "lattice":         torch.tensor(graph["lattice"],    dtype=torch.float32),
            "edge_index":      torch.tensor(graph["edge_index"], dtype=torch.long),
            "edge_offset":     torch.tensor(graph["edge_offset"],dtype=torch.float32),
            "triplet_idx":     torch.tensor(graph["triplet_idx"],dtype=torch.long),
            "num_atoms":       graph["num_atoms"],
            "num_edges":       graph["num_edges"],
            "volume":          torch.tensor(graph["volume"],     dtype=torch.float32),
            "energy":          torch.tensor(energy,              dtype=torch.float32),
            "forces":          torch.tensor(forces,              dtype=torch.float32),
            "stress":          torch.tensor(stress,              dtype=torch.float32),
            "global_features": torch.tensor([charge, theory_id],
                                            dtype=torch.float32),
            "idx":             idx,
        }

        if self.cache_dir is not None:
            torch.save(data, cache_path)

        return data


# ---------------------------------------------------------------------------
#  Collate – batches variable-size graphs into one big disconnected graph
# ---------------------------------------------------------------------------

def collate_fn(batch):
    """
    Collate a list of graph dicts into a single batched dict.

    Atom tensors are concatenated along dim 0.
    Edge indices are offset so they point into the concatenated atom array.
    Triplet indices are offset so they point into the concatenated edge array.
    A ``batch`` tensor maps each atom to its structure index.
    """
    batch_size = len(batch)
    atom_offset = 0
    edge_offset_count = 0   # running count of edges for triplet offsetting

    all_atom_types, all_pos, all_lattice = [], [], []
    all_edge_index, all_edge_offset, all_triplet_idx, all_batch = [], [], [], []
    all_energy, all_forces, all_stress = [], [], []
    all_num_atoms, all_volume = [], []
    all_global_features = []
    all_idx = []

    for i, d in enumerate(batch):
        n = d["num_atoms"]
        n_edges = d["num_edges"]

        all_atom_types.append(d["atom_types"])
        all_pos.append(d["pos"])
        all_lattice.append(d["lattice"])

        # offset edge indices by cumulative atom count
        ei = d["edge_index"].clone()
        ei += atom_offset
        all_edge_index.append(ei)

        all_edge_offset.append(d["edge_offset"])

        # offset triplet indices by cumulative edge count
        ti = d["triplet_idx"].clone()
        ti += edge_offset_count
        all_triplet_idx.append(ti)

        all_batch.append(torch.full((n,), i, dtype=torch.long))

        all_energy.append(d["energy"])
        all_forces.append(d["forces"])
        all_stress.append(d["stress"])
        all_num_atoms.append(n)
        all_volume.append(d["volume"])
        all_global_features.append(d["global_features"])
        all_idx.append(d["idx"])

        atom_offset += n
        edge_offset_count += n_edges

    return {
        "atom_types":      torch.cat(all_atom_types),
        "pos":             torch.cat(all_pos),
        "lattice":         torch.stack(all_lattice),
        "edge_index":      torch.cat(all_edge_index, dim=1),
        "edge_offset":     torch.cat(all_edge_offset),
        "triplet_idx":     torch.cat(all_triplet_idx, dim=1),
        "batch":           torch.cat(all_batch),
        "energy":          torch.stack(all_energy),
        "forces":          torch.cat(all_forces),
        "stress":          torch.stack(all_stress),
        "num_atoms":       torch.tensor(all_num_atoms, dtype=torch.long),
        "volume":          torch.stack(all_volume),
        "global_features": torch.stack(all_global_features),
        "idx":             torch.tensor(all_idx, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
#  Convenience: build train / val / test loaders
# ---------------------------------------------------------------------------

def create_dataloaders(
    csv_path: str,
    batch_size: int = 8,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    cutoff: float = 5.0,
    max_neighbors: int = 12,
    cache_dir=None,
    stress_unit: str = "kbar",
    seed: int = 42,
    num_workers: int = 0,
    fraction: float = 1.0,
):
    """Split CSV into train / val / test and return DataLoaders.

    Parameters
    ----------
    fraction : float
        Fraction of the full dataset to use (0 < fraction <= 1).
        Useful for quick experiments on large CSVs.
    """
    ds = ForceFieldDataset(
        csv_path, cutoff=cutoff, max_neighbors=max_neighbors,
        cache_dir=cache_dir, stress_unit=stress_unit,
    )

    g = torch.Generator().manual_seed(seed)
    random.seed(seed)

    n = len(ds)
    if fraction < 1.0:
        n_use = max(1, int(fraction * n))
        indices = torch.randperm(n, generator=g).tolist()[:n_use]
        ds = torch.utils.data.Subset(ds, indices)
        n = n_use
        g = torch.Generator().manual_seed(seed)

    n_train = int(train_ratio * n)
    n_val   = int(val_ratio * n)
    n_test  = n - n_train - n_val

    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        ds, [n_train, n_val, n_test], generator=g,
    )

    make = lambda d, shuffle: DataLoader(
        d, batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate_fn, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return make(train_ds, True), make(val_ds, False), make(test_ds, False)
