"""
predict.py
Inference script for the trained DefectNet Force Field.

Usage
-----
  # Predict on a CSV (only needs Structure, Charge, LevelOfTheory columns)
  python predict.py --checkpoint trained_model/best.pt \
                    --csv merged_data.csv \
                    --fraction 0.01 \
                    --out predictions.csv

  # Predict a single structure from a CIF / POSCAR file
  python predict.py --checkpoint trained_model/best.pt \
                    --structure POSCAR --charge 0 --theory HSE
"""

import argparse
import json
import random

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Structure

from dataset import collate_fn, KBAR_TO_EV_PER_A3, THEORY_MAP
from graph_utils import build_crystal_graph
from model import DefectNetForceField


def parse_args():
    p = argparse.ArgumentParser(description="DefectNet-FF Prediction")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--csv", default=None,
                   help="CSV with Structure, Charge, LevelOfTheory columns")
    p.add_argument("--structure", default=None,
                   help="Single structure file (CIF, POSCAR, or JSON)")
    p.add_argument("--out", default="predictions.csv",
                   help="Output CSV path (only for --csv mode)")
    p.add_argument("--fraction", type=float, default=1.0,
                   help="Fraction of CSV rows to predict (0<f<=1)")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    # global features for single structure prediction
    p.add_argument("--charge", type=float, default=0,
                   help="Charge for single structure (default: 0)")
    p.add_argument("--theory", default="hse",
                   help="Level of theory for single structure (default: hse)")
    return p.parse_args()


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    model = DefectNetForceField(
        atom_fea_len=cfg.get("atom_fea_len", 64),
        num_conv=cfg.get("num_conv", 4),
        num_gaussians=cfg.get("num_gaussians", 80),
        cutoff=cfg.get("cutoff", 5.0),
    )

    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()
    return model, cfg


def _build_data_dict(structure, charge, theory_str, cutoff, max_neighbors):
    """Build a single-graph data dict from a pymatgen Structure."""
    graph = build_crystal_graph(structure, cutoff=cutoff,
                                max_neighbors=max_neighbors)
    theory_id = float(THEORY_MAP.get(theory_str.strip().lower(), 0))

    return {
        "atom_types":      torch.tensor(graph["atom_types"], dtype=torch.long),
        "pos":             torch.tensor(graph["positions"],  dtype=torch.float32),
        "lattice":         torch.tensor(graph["lattice"],    dtype=torch.float32),
        "edge_index":      torch.tensor(graph["edge_index"], dtype=torch.long),
        "edge_offset":     torch.tensor(graph["edge_offset"],dtype=torch.float32),
        "triplet_idx":     torch.tensor(graph["triplet_idx"],dtype=torch.long),
        "num_atoms":       graph["num_atoms"],
        "num_edges":       graph["num_edges"],
        "volume":          torch.tensor(graph["volume"],     dtype=torch.float32),
        "global_features": torch.tensor([charge, theory_id],
                                        dtype=torch.float32),
    }


def _collate_predict(batch):
    """Collate for prediction (no energy/forces/stress targets)."""
    batch_size = len(batch)
    atom_offset = 0
    edge_offset_count = 0

    all_atom_types, all_pos, all_lattice = [], [], []
    all_edge_index, all_edge_offset, all_triplet_idx, all_batch = [], [], [], []
    all_num_atoms, all_volume = [], []
    all_global_features = []

    for i, d in enumerate(batch):
        n = d["num_atoms"]
        n_edges = d["num_edges"]

        all_atom_types.append(d["atom_types"])
        all_pos.append(d["pos"])
        all_lattice.append(d["lattice"])

        ei = d["edge_index"].clone()
        ei += atom_offset
        all_edge_index.append(ei)
        all_edge_offset.append(d["edge_offset"])

        ti = d["triplet_idx"].clone()
        ti += edge_offset_count
        all_triplet_idx.append(ti)

        all_batch.append(torch.full((n,), i, dtype=torch.long))
        all_num_atoms.append(n)
        all_volume.append(d["volume"])
        all_global_features.append(d["global_features"])

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
        "num_atoms":       torch.tensor(all_num_atoms, dtype=torch.long),
        "volume":          torch.stack(all_volume),
        "global_features": torch.stack(all_global_features),
    }


def predict_structure(model, structure, device, cutoff=5.0, max_neighbors=12,
                      charge=0.0, theory="hse"):
    """Predict energy, forces, stress for a single pymatgen Structure."""
    d = _build_data_dict(structure, charge, theory, cutoff, max_neighbors)

    # add batch dimension
    data = {
        "atom_types":      d["atom_types"],
        "pos":             d["pos"],
        "lattice":         d["lattice"].unsqueeze(0),
        "edge_index":      d["edge_index"],
        "edge_offset":     d["edge_offset"],
        "triplet_idx":     d["triplet_idx"],
        "batch":           torch.zeros(d["num_atoms"], dtype=torch.long),
        "num_atoms":       torch.tensor([d["num_atoms"]], dtype=torch.long),
        "volume":          d["volume"].unsqueeze(0),
        "global_features": d["global_features"].unsqueeze(0),
    }
    data = {k: v.to(device) for k, v in data.items()}

    with torch.enable_grad():
        pred = model(data, compute_forces=True, compute_stress=True)

    energy = pred["energy"].item()
    forces = pred["forces"].detach().cpu().numpy()
    stress = pred["stress"].detach().cpu().numpy()[0]  # (3, 3)

    return energy, forces, stress


def predict_csv(args, model, cfg, device):
    """Run predictions on a CSV and save results.

    The CSV only needs: Structure, Charge, LevelOfTheory.
    No DFT reference (Energy, Forces, Stress) required.
    """
    cutoff = cfg.get("cutoff", 5.0)
    max_neighbors = cfg.get("max_neighbors", 12)

    print(f"Reading {args.csv} …")
    orig_df = pd.read_csv(args.csv)
    n_total = len(orig_df)

    # subsample if fraction < 1
    if args.fraction < 1.0:
        n_use = max(1, int(args.fraction * n_total))
        random.seed(args.seed)
        indices = sorted(random.sample(range(n_total), n_use))
    else:
        indices = list(range(n_total))

    print(f"  {len(indices)} / {n_total} structures selected "
          f"(fraction={args.fraction})")

    # build graph dicts for selected rows
    graphs = []
    for count, idx in enumerate(indices):
        row = orig_df.iloc[idx]
        structure = Structure.from_dict(json.loads(row["Structure"]))
        charge = float(row["Charge"])
        theory = str(row["LevelOfTheory"])

        d = _build_data_dict(structure, charge, theory, cutoff, max_neighbors)
        graphs.append((idx, d))

        if (count + 1) % 100 == 0 or count + 1 == len(indices):
            print(f"  Built graph {count + 1}/{len(indices)}", end="\r")
    print()

    # predict in batches
    rows = []
    for batch_start in range(0, len(graphs), args.batch):
        batch_items = graphs[batch_start:batch_start + args.batch]
        batch_indices = [item[0] for item in batch_items]
        batch_dicts = [item[1] for item in batch_items]

        data = _collate_predict(batch_dicts)
        data = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in data.items()}

        with torch.enable_grad():
            pred = model(data, compute_forces=True, compute_stress=True)

        e_pred = pred["energy"].detach().cpu().numpy()
        f_pred = pred["forces"].detach().cpu().numpy()
        s_pred = pred["stress"].detach().cpu().numpy()
        natoms = data["num_atoms"].cpu().numpy()

        atom_cursor = 0
        for i in range(len(natoms)):
            n = natoms[i]
            csv_i = batch_indices[i]
            orig_row = orig_df.iloc[csv_i]

            fp = f_pred[atom_cursor:atom_cursor + n].tolist()
            sp = s_pred[i].tolist()

            rows.append({
                "csv_idx":          csv_i,
                "Structure":        orig_row["Structure"],
                "Charge":           orig_row["Charge"],
                "LevelOfTheory":    orig_row["LevelOfTheory"],
                "num_atoms":        int(n),
                "energy_pred":      float(e_pred[i]),
                "forces_pred":      str(fp),
                "stress_pred_eV_A3": str(sp),
            })
            atom_cursor += n

        done = min(batch_start + args.batch, len(graphs))
        print(f"  Predicted {done}/{len(graphs)}", end="\r")
    print()

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} predictions to {args.out}")


def main():
    args = parse_args()
    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )

    model, cfg = load_model(args.checkpoint, device)
    print(f"Loaded model from {args.checkpoint}  (device={device})")

    if args.structure is not None:
        # single structure prediction
        path = args.structure
        if path.endswith(".json"):
            with open(path) as f:
                structure = Structure.from_dict(json.load(f))
        else:
            structure = Structure.from_file(path)

        cutoff = cfg.get("cutoff", 5.0)
        max_neighbors = cfg.get("max_neighbors", 12)
        energy, forces, stress = predict_structure(
            model, structure, device, cutoff, max_neighbors,
            charge=args.charge, theory=args.theory,
        )

        print(f"\nEnergy : {energy:.6f} eV")
        print(f"Forces : shape {forces.shape}")
        print(forces)
        print(f"\nStress [eV/A^3]:")
        print(stress)
        print(f"Stress [kBar]:")
        print(stress / KBAR_TO_EV_PER_A3)

    elif args.csv is not None:
        predict_csv(args, model, cfg, device)

    else:
        print("Provide --csv or --structure. Use -h for help.")


if __name__ == "__main__":
    main()
