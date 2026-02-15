<div align="center">

# DefectNet

**A Graph Neural Network Force Field for Crystal Defects**

[![Force Field](https://img.shields.io/badge/Force%20Field-GNN-4CAF50?style=for-the-badge)](.)
[![Defect Modeling](https://img.shields.io/badge/Defect%20Modeling-Charge%20%2B%20Fidelity-4CAF50?style=for-the-badge)](.)
[![PyTorch](https://img.shields.io/badge/PyTorch-From%20Scratch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

*Predict DFT energy, forces, and stress for crystal structures with point defects.*
*Written from scratch in PyTorch*

</div>

---

## Overview

**DefectNet** uses **2-body** (pairwise radial) and **3-body** (angular triplet) message-passing, conditioned on global features:

| Global Feature | Description |
|:---:|---|
| **Charge state** | Defect charge (e.g., -2, -1, 0, +1, +2) |
| **DFT fidelity** | Functional level of theory (HSE, PBE, PBEsol, SCAN, LDA) |

This allows training **a single model** across mixed charge states and mixed levels of theory.

---

## Architecture

```
Input: periodic crystal structure + charge + level of theory
                            |
                            v
  1. Atom embedding           Z  -->  h_i          (learnable, indexed by atomic number)
  2. Global feature proj      [charge, theory]  --> broadcast to all atoms
  3. Gaussian distance exp    d  -->  e_ij         (80 radial basis functions)
  4. Smooth cosine cutoff     d  -->  w_ij         (ensures force continuity)
                            |
                            v
  5. N x interaction blocks:
       a) 2-body DefectNet conv    (pairwise radial message-passing)
       b) 3-body angular conv      (triplet angle message-passing)
                            |
                            v
  6. Per-atom energy head     h_i --> epsilon_i
  7. Total energy             E = sum(epsilon_i)   (extensive, sum pooling)
  8. Forces                   F = -dE/dr           (autograd)
  9. Stress                   sigma = (1/V) dE/de  (strain-derivative method)
```

<details>
<summary><b>Key design choices</b></summary>

- **Gated convolutions** -- sigmoid gating + softplus activation (CGCNN-style)
- **Cosine cutoff** -- smooth force continuity at the neighbour boundary
- **Angular basis** -- cos(theta) expanded into Gaussian basis over [-1, +1] for 3-body terms
- **Strain-derivative stress** -- computed analytically via autograd at epsilon = 0
- **Global conditioning** -- charge + theory projected into atom feature space before message passing

</details>

---

## Installation

```bash
# Create conda environment
conda create -n defectnet python=3.10 -y
conda activate defectnet
```

<table>
<tr>
<th>GPU (CUDA)</th>
<th>CPU only</th>
</tr>
<tr>
<td>

```bash
# CUDA 11.8
pip install torch --index-url \
  https://download.pytorch.org/whl/cu118

# -- OR -- CUDA 12.1
pip install torch --index-url \
  https://download.pytorch.org/whl/cu121

# Remaining dependencies
pip install pymatgen pandas ase "numpy<2"
```

</td>
<td>

```bash
# CPU-only (smaller download)
pip install torch --index-url \
  https://download.pytorch.org/whl/cpu

# Remaining dependencies
pip install pymatgen pandas ase "numpy<2"
```

</td>
</tr>
</table>

---

## Quick Start

```bash
# Train on 1% of data for a quick test (3 epochs, CPU)
python train.py --csv data.csv --fraction 0.01 --epochs 3 --batch 4 --device cpu

# Train on full dataset (GPU recommended)
python train.py --csv data.csv --epochs 200 --batch 8 --lr 1e-3 --device cuda

# Resume from checkpoint
python train.py --csv data.csv --restart trained_model/last.pt
```

```bash
# Predict on new structures (no DFT reference needed)
python predict.py --checkpoint trained_model/best.pt --csv new_structures.csv --fraction 0.1

# Predict a single structure
python predict.py --checkpoint trained_model/best.pt --structure POSCAR --charge 0 --theory HSE
```

```bash
# Relax a structure (positions only)
python optimize.py --checkpoint trained_model/best.pt --structure POSCAR --charge 0 --theory HSE

# Full relaxation (cell + positions)
python optimize.py --checkpoint trained_model/best.pt --structure POSCAR --relax_cell --fmax 0.01

# Batch relax from CSV
python optimize.py --checkpoint trained_model/best.pt --csv data.csv --fraction 0.01 --out relaxed.csv
```

---

## Input CSV Format

### Training CSV

The training CSV requires these columns:

| Column | Type | Description |
|:-------|:-----|:------------|
| `Structure` | str | JSON-serialised pymatgen Structure |
| `Energy` | float | Total DFT energy [eV] |
| `Forces` | str | Python list-of-lists `[[fx,fy,fz], ...]` [eV/A] |
| `Stress` | str | 3x3 Python list-of-lists (VASP convention) [kBar] |
| `Charge` | float | System charge (e.g., -2, -1, 0, +1, +2) |
| `LevelOfTheory` | str | DFT functional: `"HSE"`, `"PBE"`, `"PBEsol"`, `"SCAN"`, or `"LDA"` |

<details>
<summary>Optional columns (metadata only, not used by model)</summary>

| Column | Type | Description |
|:-------|:-----|:------------|
| `Directory` | str | Identifier / path |
| `Frequency` | int | Sampling weight |
| `CFE` | float | Cohesive / formation energy |
| `Tag` | str | e.g. `"bulk"`, `"defect"` |

</details>

### Prediction CSV

For inference, the CSV only needs **three columns**:

| Column | Type | Description |
|:-------|:-----|:------------|
| `Structure` | str | JSON-serialised pymatgen Structure |
| `Charge` | float | System charge |
| `LevelOfTheory` | str | DFT functional |

No DFT reference data (Energy, Forces, Stress) required.

<details>
<summary>Example row</summary>

```
Structure,Energy,Forces,Stress,Charge,LevelOfTheory
"{""@module"":""pymatgen.core.structure"",...}",-301.49,"[[0.01,-0.02,0.03],...]","[[-1.2,0,0],[0,-1.3,0],[0,0,-1.1]]",0,HSE
```

</details>

### Global Feature Encoding

| Feature | Encoding |
|:--------|:---------|
| Charge | Raw float value (e.g., -2, -1, 0, +1, +2) |
| LevelOfTheory | `HSE` = 0, `PBE` = 1, `PBEsol` = 2, `SCAN` = 3, `LDA` = 4 |

Global features are projected into the atom feature space and added to atom embeddings before message passing, so the model is conditioned on charge and fidelity throughout.

---

## Training

<details>
<summary><b>CLI Arguments</b></summary>

| Argument | Default | Description |
|:---------|:--------|:------------|
| `--csv` | (required) | Path to input CSV |
| `--fraction` | 1.0 | Fraction of data to use (e.g. 0.01 = 1%) |
| `--epochs` | 200 | Number of training epochs |
| `--batch` | 4 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--device` | cuda | Device (`cuda` or `cpu`) |
| `--cutoff` | 5.0 | Neighbour cutoff in Angstrom |
| `--max_neighbors` | 12 | Max neighbours per atom |
| `--num_conv` | 4 | Number of interaction blocks |
| `--atom_fea_len` | 64 | Atom embedding dimension |
| `--num_gaussians` | 80 | Radial Gaussian basis functions |
| `--weight_energy` | 1.0 | Energy loss weight |
| `--weight_force` | 50.0 | Force loss weight |
| `--weight_stress` | 0.1 | Stress loss weight |
| `--train_ratio` | 0.8 | Train split ratio |
| `--val_ratio` | 0.1 | Validation split ratio |
| `--seed` | 42 | Random seed |
| `--outdir` | trained_model | Output directory |
| `--restart` | None | Checkpoint to resume from |
| `--cache_dir` | None | Directory for caching processed graphs |
| `--stress_unit` | kbar | Stress unit in CSV (`kbar` or `ev_per_a3`) |

</details>

### Loss Function

Multi-task loss with configurable weights:

```
L = w_e * MSE(E/N) + w_f * MSE(F) + w_s * MSE(sigma)
```

- Energy is normalised **per atom** before computing MSE
- Forces are compared per-component (Fx, Fy, Fz)
- Stress is compared per-component of the 3x3 tensor

### Training Output

```
trained_model/
  best.pt                    # Best model checkpoint (lowest val loss)
  last.pt                    # Last epoch checkpoint (for resuming)
  config.json                # Training configuration
  train_predictions.csv      # Predicted vs DFT for training set
  val_predictions.csv        # Predicted vs DFT for validation set
  test_predictions.csv       # Predicted vs DFT for test set
```

<details>
<summary><b>Training prediction CSV columns</b></summary>

| Column | Description |
|:-------|:------------|
| `csv_idx` | Row index in the original CSV file |
| `Structure` | JSON-serialised pymatgen Structure |
| `Charge` | System charge |
| `LevelOfTheory` | DFT functional |
| `num_atoms` | Number of atoms in the structure |
| `energy_dft` | DFT total energy [eV] |
| `energy_pred` | Model-predicted total energy [eV] |
| `energy_err_per_atom` | (predicted - DFT) / num_atoms [eV/atom] |
| `forces_dft` | DFT forces [eV/A] |
| `forces_pred` | Predicted forces [eV/A] |
| `stress_dft_eV_A3` | DFT stress tensor 3x3 [eV/A^3] |
| `stress_pred_eV_A3` | Predicted stress tensor 3x3 [eV/A^3] |

</details>

---

## Prediction / Inference

<details>
<summary><b>CLI Arguments</b></summary>

| Argument | Default | Description |
|:---------|:--------|:------------|
| `--checkpoint` | (required) | Path to `.pt` checkpoint |
| `--csv` | None | CSV with Structure, Charge, LevelOfTheory columns |
| `--structure` | None | Single structure file (CIF/POSCAR/JSON) |
| `--out` | predictions.csv | Output CSV path (batch mode) |
| `--fraction` | 1.0 | Fraction of CSV rows to predict (e.g. 0.01 = 1%) |
| `--batch` | 4 | Batch size |
| `--device` | cuda | Device |
| `--seed` | 42 | Random seed for fraction sampling |
| `--charge` | 0 | Charge for single structure mode |
| `--theory` | hse | Level of theory for single structure mode |

</details>

<details>
<summary><b>Prediction output CSV columns</b></summary>

| Column | Description |
|:-------|:------------|
| `csv_idx` | Row index in the original CSV |
| `Structure` | JSON-serialised pymatgen Structure |
| `Charge` | System charge |
| `LevelOfTheory` | DFT functional |
| `num_atoms` | Number of atoms |
| `energy_pred` | Predicted total energy [eV] |
| `forces_pred` | Predicted forces [eV/A] |
| `stress_pred_eV_A3` | Predicted stress tensor 3x3 [eV/A^3] |

</details>

---

## Geometry Optimisation

`optimize.py` wraps the trained DefectNet model as an [ASE Calculator](https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html), so you can use any ASE optimiser (BFGS, LBFGS, FIRE) to relax crystal structures with the ML force field.

### How it works

```
                 ASE Optimiser (BFGS / LBFGS / FIRE)
                          |
                          v
               DefectNetCalculator (ASE Calculator)
                 |         |          |
              energy    forces     stress      <-- from trained model
                          |
                          v
               Update positions (and cell if --relax_cell)
                          |
                     Converged?  -->  relaxed structure
```

The `DefectNetCalculator` class can also be used directly in Python:

```python
from optimize import DefectNetCalculator
from ase.io import read
from ase.optimize import BFGS

atoms = read("POSCAR")
calc = DefectNetCalculator(
    checkpoint="trained_model/best.pt",
    charge=0, theory="hse", device="cpu",
)
atoms.calc = calc

opt = BFGS(atoms)
opt.run(fmax=0.05, steps=200)

print(f"Relaxed energy: {atoms.get_potential_energy():.4f} eV")
```

<details>
<summary><b>CLI Arguments</b></summary>

| Argument | Default | Description |
|:---------|:--------|:------------|
| `--checkpoint` | (required) | Path to `.pt` checkpoint |
| `--structure` | None | Single structure file (POSCAR/CIF/JSON) |
| `--csv` | None | CSV with Structure, Charge, LevelOfTheory columns |
| `--fraction` | 1.0 | Fraction of CSV rows to relax |
| `--charge` | 0 | Charge state (single structure mode) |
| `--theory` | hse | Level of theory (single structure mode) |
| `--fmax` | 0.05 | Force convergence threshold [eV/A] |
| `--steps` | 200 | Max optimisation steps |
| `--optimizer` | bfgs | ASE optimiser (`bfgs`, `lbfgs`, `fire`) |
| `--relax_cell` | false | Relax cell parameters + positions |
| `--trajectory` | None | Save ASE `.traj` file |
| `--out` | relaxed.csv | Output CSV (batch mode) |
| `--out_structure` | None | Save relaxed structure file (single mode) |
| `--device` | cpu | Device (`cpu` or `cuda`) |

</details>

<details>
<summary><b>Batch relaxation output CSV columns</b></summary>

| Column | Description |
|:-------|:------------|
| `csv_idx` | Row index in the original CSV |
| `Structure_relaxed` | JSON-serialised relaxed pymatgen Structure |
| `Charge` | System charge |
| `LevelOfTheory` | DFT functional |
| `num_atoms` | Number of atoms |
| `energy_relaxed` | Relaxed total energy [eV] |
| `forces_relaxed` | Final forces [eV/A] |
| `max_force` | Maximum force magnitude [eV/A] |
| `converged` | Whether optimisation converged within `--steps` |

</details>

---

## Project Structure

```
DefectNet/
  model.py          # DefectNetForceField (2-body + 3-body GNN)
  graph_utils.py    # Crystal graph construction from pymatgen Structure
  dataset.py        # Dataset, collate function, DataLoader creation
  train.py          # Multi-task training loop with prediction CSV saving
  predict.py        # Inference script (batch CSV or single structure)
  optimize.py       # Geometry optimisation via ASE + trained model
  data.csv          # Sample dataset (8,221 structures, mixed charge & theory)
  README.md         # This file
```

| Module | Description |
|:-------|:------------|
| `model.py` | `DefectNetForceField` with GaussianSmearing, CosineCutoff, AngularBasis, DefectNetConv (2-body), ThreeBodyConv (3-body), global conditioning, energy head, autograd forces + stress |
| `graph_utils.py` | `build_crystal_graph()` with PBC-aware neighbour search, edge offsets, and triplet index construction |
| `dataset.py` | `ForceFieldDataset` with on-the-fly graph building (optional disk cache), `collate_fn` for variable-size graphs, train/val/test splitting |
| `train.py` | AdamW + ReduceLROnPlateau, gradient clipping, best/last checkpointing, per-split prediction CSVs |
| `predict.py` | Inference on batch CSV or single structure file, with fraction support |
| `optimize.py` | ASE Calculator wrapper (`DefectNetCalculator`) + geometry optimisation with BFGS/LBFGS/FIRE, single or batch mode, optional cell relaxation |

---

## Units

| Quantity | Unit |
|:---------|:-----|
| Energy | eV |
| Forces | eV/A |
| Stress (internal) | eV/A^3 |
| Stress (VASP convention) | kBar (1 eV/A^3 = 1602.1766 kBar) |

---

## Authors

**Md Habibur Rahman** and **Arun Mannodi-Kanakkithodi**

School of Materials Engineering, Purdue University, West Lafayette, IN 47907, USA

Contact: [rahma103@purdue.edu](mailto:rahma103@purdue.edu)

January 2025

---

## References

If you use DefectNet, please cite:

> **Accelerating defect predictions in semiconductors using graph neural networks**
> Md Habibur Rahman, Prince Gollapalli, Panayotis Manganaris, Satyesh Kumar Yadav, Ghanshyam Pilania, Brian DeCost, Kamal Choudhary, and Arun Mannodi-Kanakkithodi,
> *APL Mach. Learn.* **2**, 016122 (2024).
> [DOI: 10.1063/5.0176333](https://doi.org/10.1063/5.0176333)

> **DeFecT-FF: Accelerated Modeling of Defects in Cd-Zn-Te-Se-S Compounds Combining High-Throughput DFT and Machine Learning Force Fields**
> Md Habibur Rahman and Arun Mannodi-Kanakkithodi,
> *arXiv* preprint arXiv:2510.23514 (2025).
> [arXiv: 2510.23514](https://arxiv.org/abs/2510.23514)

### Related works

1. **CGCNN** -- Crystal Graph Convolutional Neural Networks
   T. Xie and J. C. Grossman, *Phys. Rev. Lett.* **120**, 145301 (2018).
   [DOI: 10.1103/PhysRevLett.120.145301](https://doi.org/10.1103/PhysRevLett.120.145301)

2. **ALIGNN** -- Atomistic Line Graph Neural Network
   K. Choudhary and B. DeCost, *npj Comput. Mater.* **7**, 185 (2021).
   [DOI: 10.1038/s41524-021-00650-1](https://doi.org/10.1038/s41524-021-00650-1)

3. **M3GNet** -- Universal Graph Neural Network Interatomic Potential
   C. Chen and S. P. Ong, *Nat. Comput. Sci.* **2**, 718--728 (2022).
   [DOI: 10.1038/s43588-022-00349-3](https://doi.org/10.1038/s43588-022-00349-3)

4. **CHGNet** -- Pretrained Universal Neural Network Potential with Charge Informed
   B. Deng *et al.*, *Nat. Mach. Intell.* **5**, 1031--1041 (2023).
   [DOI: 10.1038/s42256-023-00716-3](https://doi.org/10.1038/s42256-023-00716-3)

5. **MACE** -- Higher Order Equivariant Message Passing Neural Networks
   I. Batatia *et al.*, *Advances in Neural Information Processing Systems* **35** (NeurIPS 2022).
   [arXiv: 2206.07697](https://arxiv.org/abs/2206.07697)

6. **SchNet** -- A Continuous-Filter Convolutional Neural Network for Modeling Quantum Interactions
   K. T. Schutt *et al.*, *Advances in Neural Information Processing Systems* **30** (NeurIPS 2017).
   [DOI: 10.48550/arXiv.1706.08566](https://arxiv.org/abs/1706.08566)

7. **DimeNet++** -- Fast and Uncertainty-Aware Directional Message Passing
   J. Gasteiger, S. Giri, J. T. Margraf, and S. Gunnemann, *ICLR 2020 Workshop*.
   [arXiv: 2011.14115](https://arxiv.org/abs/2011.14115)

---

## License

MIT License -- free to use, modify, and distribute.
