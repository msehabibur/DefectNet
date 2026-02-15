"""
optimize.py
Geometry optimisation using a trained DefectNet model as an ASE calculator.

Usage
-----
  # Relax a single structure (POSCAR / CIF / JSON)
  python optimize.py --checkpoint trained_model/best.pt --structure POSCAR \
                     --charge 0 --theory HSE --device cpu

  # Relax with tighter convergence
  python optimize.py --checkpoint trained_model/best.pt --structure POSCAR \
                     --fmax 0.01 --steps 500

  # Relax cell + positions (full optimisation)
  python optimize.py --checkpoint trained_model/best.pt --structure POSCAR \
                     --relax_cell

  # Batch relax from CSV
  python optimize.py --checkpoint trained_model/best.pt --csv data.csv \
                     --fraction 0.01 --out relaxed.csv
"""

import argparse
import json
import random
import warnings

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import ExpCellFilter
from ase.optimize import BFGS, FIRE, LBFGS

from dataset import THEORY_MAP
from graph_utils import build_crystal_graph
from model import DefectNetForceField


# ---------------------------------------------------------------------------
#  ASE Calculator
# ---------------------------------------------------------------------------

class DefectNetCalculator(Calculator):
    """
    ASE Calculator wrapping a trained DefectNetForceField model.

    Provides energy, forces, and stress to any ASE optimiser or
    dynamics driver.

    Parameters
    ----------
    checkpoint : str
        Path to a DefectNet .pt checkpoint file.
    charge : float
        Defect charge state.
    theory : str
        Level of theory (hse, pbe, pbesol, scan, lda).
    device : str
        'cpu' or 'cuda'.
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, checkpoint, charge=0.0, theory="hse", device="cpu",
                 **kwargs):
        super().__init__(**kwargs)
        self.device = torch.device(device)
        self.charge = float(charge)
        self.theory = theory.strip().lower()

        # load model
        ckpt = torch.load(checkpoint, map_location=self.device,
                          weights_only=False)
        cfg = ckpt.get("config", {})
        self.cutoff = cfg.get("cutoff", 5.0)
        self.max_neighbors = cfg.get("max_neighbors", 12)

        self.model = DefectNetForceField(
            atom_fea_len=cfg.get("atom_fea_len", 64),
            num_conv=cfg.get("num_conv", 4),
            num_gaussians=cfg.get("num_gaussians", 80),
            cutoff=self.cutoff,
        )
        state = ckpt["model_state"] if "model_state" in ckpt else ckpt
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if properties is None:
            properties = self.implemented_properties
        super().calculate(atoms, properties, system_changes)

        # convert ASE Atoms → pymatgen Structure
        structure = AseAtomsAdaptor.get_structure(self.atoms)

        # build graph
        graph = build_crystal_graph(structure, cutoff=self.cutoff,
                                    max_neighbors=self.max_neighbors)
        theory_id = float(THEORY_MAP.get(self.theory, 0))

        # assemble data dict (single structure, batch_size=1)
        data = {
            "atom_types":   torch.tensor(graph["atom_types"], dtype=torch.long),
            "pos":          torch.tensor(graph["positions"], dtype=torch.float32),
            "edge_index":   torch.tensor(graph["edge_index"], dtype=torch.long),
            "edge_offset":  torch.tensor(graph["edge_offset"], dtype=torch.float32),
            "triplet_idx":  torch.tensor(graph["triplet_idx"], dtype=torch.long),
            "batch":        torch.zeros(graph["num_atoms"], dtype=torch.long),
            "num_atoms":    torch.tensor([graph["num_atoms"]], dtype=torch.long),
            "volume":       torch.tensor([graph["volume"]], dtype=torch.float32),
            "global_features": torch.tensor([[self.charge, theory_id]],
                                            dtype=torch.float32),
        }
        data = {k: v.to(self.device) for k, v in data.items()}

        with torch.enable_grad():
            pred = self.model(data, compute_forces=True, compute_stress=True)

        energy = pred["energy"].item()
        forces = pred["forces"].detach().cpu().numpy()
        # model stress is (1, 3, 3) in eV/A^3; ASE expects Voigt (6,) in eV/A^3
        stress_3x3 = pred["stress"].detach().cpu().numpy()[0]
        # ASE convention: positive stress = compressive
        # Voigt order: xx, yy, zz, yz, xz, xy
        stress_voigt = np.array([
            stress_3x3[0, 0], stress_3x3[1, 1], stress_3x3[2, 2],
            stress_3x3[1, 2], stress_3x3[0, 2], stress_3x3[0, 1],
        ])

        self.results["energy"] = energy
        self.results["forces"] = forces
        self.results["stress"] = stress_voigt


# ---------------------------------------------------------------------------
#  Optimiser map
# ---------------------------------------------------------------------------

OPTIMIZERS = {
    "bfgs": BFGS,
    "lbfgs": LBFGS,
    "fire": FIRE,
}


# ---------------------------------------------------------------------------
#  Single structure relaxation
# ---------------------------------------------------------------------------

def relax_structure(atoms, calc, fmax=0.05, steps=200, relax_cell=False,
                    optimizer="bfgs", trajectory=None):
    """
    Relax an ASE Atoms object using a DefectNetCalculator.

    Parameters
    ----------
    atoms : ase.Atoms
    calc : DefectNetCalculator
    fmax : float
        Force convergence threshold [eV/A].
    steps : int
        Maximum optimisation steps.
    relax_cell : bool
        If True, relax both cell and positions (via ExpCellFilter).
    optimizer : str
        'bfgs', 'lbfgs', or 'fire'.
    trajectory : str or None
        Path to save ASE trajectory file.

    Returns
    -------
    atoms : ase.Atoms (relaxed)
    converged : bool
    """
    atoms.calc = calc

    opt_cls = OPTIMIZERS.get(optimizer.lower(), BFGS)

    if relax_cell:
        opt_target = ExpCellFilter(atoms)
    else:
        opt_target = atoms

    opt = opt_cls(opt_target, trajectory=trajectory)
    converged = opt.run(fmax=fmax, steps=steps)

    return atoms, converged


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Geometry optimisation with trained DefectNet force field",
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to .pt checkpoint")
    # input
    p.add_argument("--structure", default=None,
                   help="Single structure file (POSCAR, CIF, JSON)")
    p.add_argument("--csv", default=None,
                   help="CSV with Structure, Charge, LevelOfTheory columns")
    p.add_argument("--fraction", type=float, default=1.0,
                   help="Fraction of CSV rows to relax")
    # global features
    p.add_argument("--charge", type=float, default=0,
                   help="Charge state (single structure mode)")
    p.add_argument("--theory", default="hse",
                   help="Level of theory (single structure mode)")
    # optimiser settings
    p.add_argument("--fmax", type=float, default=0.05,
                   help="Force convergence [eV/A]")
    p.add_argument("--steps", type=int, default=200,
                   help="Max optimisation steps")
    p.add_argument("--optimizer", default="bfgs",
                   choices=["bfgs", "lbfgs", "fire"],
                   help="ASE optimiser")
    p.add_argument("--relax_cell", action="store_true",
                   help="Relax cell parameters + positions")
    p.add_argument("--trajectory", default=None,
                   help="Save ASE trajectory to this file (.traj)")
    # output
    p.add_argument("--out", default="relaxed.csv",
                   help="Output CSV (batch mode)")
    p.add_argument("--out_structure", default=None,
                   help="Save relaxed structure (CIF/POSCAR, single mode)")
    # device
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    calc = DefectNetCalculator(
        checkpoint=args.checkpoint,
        charge=args.charge,
        theory=args.theory,
        device=args.device,
    )
    print(f"Loaded DefectNet calculator (device={args.device})")

    adaptor = AseAtomsAdaptor()

    if args.structure is not None:
        # --- single structure mode ---
        path = args.structure
        if path.endswith(".json"):
            with open(path) as f:
                structure = Structure.from_dict(json.load(f))
        else:
            structure = Structure.from_file(path)

        atoms = adaptor.get_atoms(structure)
        print(f"Input: {len(atoms)} atoms, formula = {atoms.get_chemical_formula()}")
        print(f"Charge = {args.charge}, Theory = {args.theory}")
        print(f"Optimizer = {args.optimizer.upper()}, "
              f"fmax = {args.fmax} eV/A, max_steps = {args.steps}")
        if args.relax_cell:
            print("Relaxing cell + positions")
        else:
            print("Relaxing positions only (fixed cell)")

        atoms, converged = relax_structure(
            atoms, calc,
            fmax=args.fmax,
            steps=args.steps,
            relax_cell=args.relax_cell,
            optimizer=args.optimizer,
            trajectory=args.trajectory,
        )

        status = "CONVERGED" if converged else "NOT CONVERGED"
        print(f"\nOptimisation: {status}")
        print(f"Final energy : {atoms.get_potential_energy():.6f} eV")
        print(f"Max force    : {np.max(np.linalg.norm(atoms.get_forces(), axis=1)):.6f} eV/A")

        if args.out_structure:
            relaxed = adaptor.get_structure(atoms)
            relaxed.to(filename=args.out_structure)
            print(f"Relaxed structure saved to {args.out_structure}")

    elif args.csv is not None:
        # --- batch mode ---
        print(f"Reading {args.csv} ...")
        df = pd.read_csv(args.csv)
        n_total = len(df)

        if args.fraction < 1.0:
            n_use = max(1, int(args.fraction * n_total))
            random.seed(args.seed)
            indices = sorted(random.sample(range(n_total), n_use))
        else:
            indices = list(range(n_total))

        print(f"  Relaxing {len(indices)} / {n_total} structures")

        rows = []
        for count, idx in enumerate(indices):
            row = df.iloc[idx]
            structure = Structure.from_dict(json.loads(row["Structure"]))
            charge = float(row["Charge"])
            theory = str(row["LevelOfTheory"])

            # update calculator for this structure's charge/theory
            calc.charge = charge
            calc.theory = theory.strip().lower()

            atoms = adaptor.get_atoms(structure)
            atoms, converged = relax_structure(
                atoms, calc,
                fmax=args.fmax,
                steps=args.steps,
                relax_cell=args.relax_cell,
                optimizer=args.optimizer,
            )

            relaxed_structure = adaptor.get_structure(atoms)
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces().tolist()
            max_force = float(np.max(np.linalg.norm(atoms.get_forces(), axis=1)))

            rows.append({
                "csv_idx": idx,
                "Structure_relaxed": relaxed_structure.to(fmt="json"),
                "Charge": charge,
                "LevelOfTheory": theory,
                "num_atoms": len(atoms),
                "energy_relaxed": energy,
                "forces_relaxed": str(forces),
                "max_force": max_force,
                "converged": converged,
            })

            print(f"  [{count+1}/{len(indices)}] idx={idx}  "
                  f"E={energy:.4f} eV  Fmax={max_force:.4f} eV/A  "
                  f"{'OK' if converged else 'NOT CONVERGED'}")

        out_df = pd.DataFrame(rows)
        out_df.to_csv(args.out, index=False)
        print(f"\nSaved {len(out_df)} relaxed structures to {args.out}")

    else:
        print("Provide --structure or --csv. Use -h for help.")


if __name__ == "__main__":
    main()
