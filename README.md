# DefectNet

![Force Field](https://img.shields.io/badge/Force%20Field-GNN-4CAF50?style=flat-square)
![Defect Modeling](https://img.shields.io/badge/Defect%20Modeling-Charge%20%2B%20Fidelity-4CAF50?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-From%20Scratch-EE4C2C?style=flat-square)
![License: MIT](https://img.shields.io/badge/License-MIT-blue?style=flat-square)

---

## What is DefectNet?

**DefectNet** is a graph neural network force field for predicting DFT **energy**, **forces**, and **stress** of crystal structures with point defects. Written from scratch in PyTorch with no dependency on ALIGNN or other GNN libraries.

The model uses **2-body** (pairwise radial) and **3-body** (angular triplet) message-passing, conditioned on global features:

- **Charge state** of the defect (e.g., -2, -1, 0, +1, +2)
- **DFT functional fidelity** (HSE, PBE, PBEsol, SCAN, LDA)

This allows training a single model across mixed charge states and mixed levels of theory.

---

## Architecture

```
Input: periodic crystal structure + charge + level of theory

1. Atom embedding           Z  -->  h_i          (learnable, indexed by atomic number)
2. Global feature proj      [charge, theory]  --> broadcast to all atoms
3. Gaussian distance exp    d  -->  e_ij         (80 radial basis functions)
4. Smooth cosine cutoff     d  -->  w_ij         (ensures force continuity)
5. N x interaction blocks, each containing:
     a) 2-body DefectNet conv    (pairwise radial message-passing)
     b) 3-body angular conv      (triplet angle message-passing)
6. Per-atom energy head     h_i --> epsilon_i
7. Total energy             E = sum(epsilon_i)   (extensive, sum pooling)
8. Forces                   F = -dE/dr           (autograd)
9. Stress                   sigma = (1/V) dE/de  (strain-derivative method)
```

**Key design choices:**

- **Gated convolutions** with sigmoid gating and softplus activation (CGCNN-style)
- **Cosine cutoff** ensures smooth force continuity at the neighbour boundary
- **Angular basis** expands cos(theta) into Gaussian basis over [-1, +1] for 3-body terms
- **Strain-derivative stress** computed analytically via autograd at epsilon = 0
- **Global conditioning** projects charge + theory into atom feature space before message passing

---

## Installation

```bash
# Create conda environment
conda create -n defectnet python=3.10 -y
conda activate defectnet
```

### GPU (CUDA)

```bash
# Install PyTorch with CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Or for CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install pymatgen pandas "numpy<2"
```

### CPU only

```bash
# Install PyTorch (CPU-only, smaller download)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install pymatgen pandas "numpy<2"
```

---

## Quick Start

```bash
# Train on 1% of data for a quick test (3 epochs, CPU)
python train.py --csv merged_data.csv --fraction 0.01 --epochs 3 --batch 4 --device cpu

# Train on full dataset (GPU recommended)
python train.py --csv merged_data.csv --epochs 200 --batch 8 --lr 1e-3 --device cuda

# Resume from checkpoint
python train.py --csv merged_data.csv --restart trained_model/last.pt

# Predict on new structures (no DFT reference needed)
python predict.py --checkpoint trained_model/best.pt --csv new_structures.csv --fraction 0.1

# Predict a single structure
python predict.py --checkpoint trained_model/best.pt --structure POSCAR --charge 0 --theory HSE
```

---

## Input CSV Format

### Training CSV

The training CSV requires these columns:

| Column | Type | Description |
|--------|------|-------------|
| `Structure` | str | JSON-serialised pymatgen Structure |
| `Energy` | float | Total DFT energy [eV] |
| `Forces` | str | Python list-of-lists `[[fx,fy,fz], ...]` [eV/A] |
| `Stress` | str | 3x3 Python list-of-lists (VASP convention) [kBar] |
| `Charge` | float | System charge (e.g., -2, -1, 0, +1, +2) |
| `LevelOfTheory` | str | DFT functional: `"HSE"`, `"PBE"`, `"PBEsol"`, `"SCAN"`, or `"LDA"` |

Optional columns (metadata only, not used by model):

| Column | Type | Description |
|--------|------|-------------|
| `Directory` | str | Identifier / path |
| `Frequency` | int | Sampling weight |
| `CFE` | float | Cohesive / formation energy |
| `Tag` | str | e.g. `"bulk"`, `"defect"` |

### Prediction CSV

For inference, the CSV only needs **three columns**:

| Column | Type | Description |
|--------|------|-------------|
| `Structure` | str | JSON-serialised pymatgen Structure |
| `Charge` | float | System charge |
| `LevelOfTheory` | str | DFT functional |

No DFT reference data (Energy, Forces, Stress) required.

### Example Row

```
Structure,Energy,Forces,Stress,Charge,LevelOfTheory
"{""@module"":""pymatgen.core.structure"",...}",-301.49,"[[0.01,-0.02,0.03],...]","[[-1.2,0,0],[0,-1.3,0],[0,0,-1.1]]",0,HSE
```

### Global Feature Encoding

| Feature | Encoding |
|---------|----------|
| Charge | Raw float value (e.g., -2, -1, 0, +1, +2) |
| LevelOfTheory | `HSE` = 0, `PBE` = 1, `PBEsol` = 2, `SCAN` = 3, `LDA` = 4 |

Global features are projected into the atom feature space and added to atom embeddings before message passing, so the model is conditioned on charge and fidelity throughout.

---

## Training

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
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

### Loss Function

Multi-task loss with configurable weights:

```
L = w_e * MSE(E/N) + w_f * MSE(F) + w_s * MSE(sigma)
```

- Energy is normalised per atom before computing MSE
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

### Training Prediction CSV Columns

| Column | Description |
|--------|-------------|
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

---

## Prediction / Inference

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
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

### Prediction Output CSV Columns

| Column | Description |
|--------|-------------|
| `csv_idx` | Row index in the original CSV |
| `Structure` | JSON-serialised pymatgen Structure |
| `Charge` | System charge |
| `LevelOfTheory` | DFT functional |
| `num_atoms` | Number of atoms |
| `energy_pred` | Predicted total energy [eV] |
| `forces_pred` | Predicted forces [eV/A] |
| `stress_pred_eV_A3` | Predicted stress tensor 3x3 [eV/A^3] |

---

## Project Structure

```
DefectNet/
  model.py          # DefectNetForceField model (2-body + 3-body GNN)
  graph_utils.py    # Crystal graph construction from pymatgen Structure
  dataset.py        # Dataset, collate function, DataLoader creation
  train.py          # Multi-task training loop with prediction CSV saving
  predict.py        # Inference script (batch CSV or single structure)
  README.md         # This file
```

| Module | Description |
|--------|-------------|
| `model.py` | `DefectNetForceField` with GaussianSmearing, CosineCutoff, AngularBasis, DefectNetConv (2-body), ThreeBodyConv (3-body), global conditioning, energy head, autograd forces + stress |
| `graph_utils.py` | `build_crystal_graph()` from pymatgen Structure with PBC-aware neighbour search, edge offsets, and triplet index construction |
| `dataset.py` | `ForceFieldDataset` reads CSV, builds graphs on-the-fly (with optional disk cache), `collate_fn` batches variable-size graphs, `create_dataloaders` handles train/val/test splitting with fraction support |
| `train.py` | Multi-task training with AdamW + ReduceLROnPlateau, gradient clipping, best/last checkpointing, per-split prediction CSV saving |
| `predict.py` | Inference on CSV (no DFT targets needed) or single structure file, with fraction support |

---

## Units

| Quantity | Unit |
|----------|------|
| Energy | eV |
| Forces | eV/A |
| Stress (internal) | eV/A^3 |
| Stress (VASP convention) | kBar (1 eV/A^3 = 1602.1766 kBar) |

---

## About the Author

Developed by **Md Habibur Rahman**
Ph.D. Candidate, School of Materials Engineering
Purdue University

---

## License

MIT License.
Feel free to use, modify, and distribute.
