# DefectNet

DefectNet is a machine-learned interatomic potential for neutral and charged point defects in chalcogenide
semiconductors (II–VI such as CdTe, I–III–VI<sub>2</sub> such as CuInS<sub>2</sub>, I<sub>2</sub>–II–IV–VI<sub>4</sub>
such as Cu<sub>2</sub>ZnSnS<sub>4</sub>). One set of weights covers six levels of theory, each with its own output head,
and takes the total charge of the cell as an input, so a charged defect supercell is relaxed the same way as a
neutral one. The total charge enters in two places: a charge embedding in every atomic feature, and a
charge-equilibration (QEq) head that solves for atomic charges summing to the total charge and adds their periodic
electrostatic energy.

It is a fine-tune of the TECE-OAM-RRA-1.0 foundation model of the [TACE](https://github.com/xvzemin/tace)
architecture (Xu, Xie and Hu), trained on 520,660 DFT frames with total charges from −5 to +5, with TACE's QEq
electrostatics added in a fourth fine-tuning stage. This repository
gives you the model as an ASE calculator, the defect workflows built on it, and the files to fine-tune it on
your own calculations.

## Install

DefectNet runs on TACE at commit `057e463` with two small fixes (`patches/tace-057e463.patch`: the total-charge
embedding and a loss-skip counter on resumed training). Newer TACE releases changed the model code and are not
tested with these weights. `defectnet.load` also imports `defectnet/qeq.py`, which makes the QEq electrostatics of
TACE 057e463 periodic (it would otherwise treat a crystal as an open cluster); the weights were trained and tested
with it.

```bash
conda create -n defectnet python=3.13 -y && conda activate defectnet
pip install torch                       # pick the CUDA / CPU build for your machine from pytorch.org
git clone https://github.com/xvzemin/tace.git && cd tace
git checkout 057e463 && git apply ../DefectNet/patches/tace-057e463.patch
pip install -e . && cd ../DefectNet
pip install -e .
```

The weights (`DefectNet-v1.0.pt`, 891 MB, md5 `3d1f42346c153044293e6882f35b8e33`) are attached to the
[v1.0 release](../../releases/tag/v1.0) of this repository:

```bash
gh release download v1.0 -R msehabibur/DefectNet -p DefectNet-v1.0.pt
```

## Use it

```python
from ase.build import bulk
import defectnet as dn

calc = dn.load("DefectNet-v1.0.pt", functional="PBEsol")     # device: cuda, xpu or cpu, picked automatically

cell = bulk("CdTe", "zincblende", a=6.48, cubic=True).repeat(3)
del cell[0]                                                    # Cd vacancy in a 216-atom cell
relaxed = dn.relax(cell, calc, q=-2)                           # rattle, then FIRE to 0.02 eV/Å, cell fixed
print(relaxed.info["energy"], relaxed.info["converged"])
```

The charge is read from `atoms.info["total_charge"]` (q = +1 means one electron removed). `dn.relax` and
`dn.charged(atoms, q)` set it for you; if you attach the calculator yourself, set it first, because a cell without
it is treated as neutral.

Use supercells of about the size of the defect cells DefectNet was trained on (215–289 atoms). The charge enters
the features of every atom, so the energy cost of a charge depends on the cell size: in a 64-atom cell the
formation energies of charged defects come out unphysical.

| `functional` | head | level of theory of the training data |
|---|---|---|
| `"PBEsol"` | 0 | PBEsol |
| `"PBE"` | 1 | PBE |
| `"PBE+U"` | 2 | PBE+U |
| `"HSE06"` | 3 | HSE06 with PBE pseudopotentials and geometries |
| `"HSE06(PBEsol)"` | 4 | HSE06 on PBEsol (VASP `GGA = PS`) |
| `"HSE06+SOC"` | 5 | HSE06(PBEsol) with spin–orbit coupling |

What the package does:

| function | what it gives |
|---|---|
| `dn.load(model, functional, device)` | ASE calculator for one head |
| `dn.relax(atoms, calc, q)` | fixed-cell relaxation from a rattled start (10 % of the shortest bond, the ShakeNBreak default) |
| `dn.formation_energy(e_def, e_bulk, change, mu, q, vbm, e_fermi)` | E<sub>f</sub> = E<sub>def</sub> − E<sub>bulk</sub> − Σ n<sub>i</sub> μ<sub>i</sub> + q (E<sub>VBM</sub> + E<sub>F</sub>) |
| `dn.transition_levels(ef0, gap)` | charge-transition levels inside the gap, negative-U pairs included |
| `dn.vacancy_hop(pristine, site, calc, q)` | vacancy migration barrier by climbing-image NEB (IDPP start) |
| `defectnet-data` | converts your DFT runs into training frames |

Worked scripts are in `examples/`: `01_quickstart.py` (energies at each head, a charged vacancy),
`02_formation_energy.py` (V<sub>Cd</sub> in CdTe against the Fermi level with DefectNet chemical potentials) and
`03_vacancy_neb.py` (V<sub>Cd</sub> migration barrier at two charge states). `tutorial.ipynb` runs the whole path
in one notebook, with its outputs saved: install, load the weights, a charged-vacancy formation-energy diagram, a
migration barrier, a short fine-tune on the CdTe frames in `examples/data/`, and the same calls with the fine-tuned model.
`examples/paper/` has one notebook per figure of the DefectNet paper (Figs. 6a, 7, 8a, 8b and 8d), each with
its DFT data as Arrow tables and its outputs saved; `examples/paper/README.md` lists them and their run times.

DefectNet does not predict band edges. The VBM and the gap in a formation-energy diagram come from a DFT
calculation of the pristine cell at the same level of theory.

## Fine-tune it on your own system

1. Collect frames. Every ionic step of a run becomes one frame, labelled with its head and its charge. Anything
   `ase.io.read` reads works (VASP `vasprun.xml` / `OUTCAR`, Quantum ESPRESSO `pw.x` output, and others):

   ```bash
   defectnet-data --functional PBEsol --charge 0  -o train.xyz bulk/vasprun.xml
   defectnet-data --functional PBEsol --charge -1 -o train.xyz V_Cd_q-1/vasprun.xml   # appends
   ```

   Use one head per level of theory and include the pristine cells and elemental references along with the
   defects, because formation energies are differences between them. Data from a code other than VASP has a
   different energy zero for each element, so give it enough bulk and reference cells for the head to absorb the
   offset.

2. Train. Copy `finetune/finetune.yaml` next to `train.xyz` and `DefectNet-v1.0.pt`, then

   ```bash
   defectnet-train -cn finetune.yaml
   ```

   `defectnet-train` is `tace-train` with the periodic QEq electrostatics registered first, and takes the same
   arguments.

   The template runs on one GPU for 2,000 steps with learning rate 1e-5 and takes 10 % of the frames for
   validation (with fewer than 10 frames that leaves none, so raise `dataset.valid_ratio` or give a
   `dataset.valid_file`). Any setting can also be changed on the command line, for example
   `defectnet-train -cn finetune.yaml trainer.max_steps=5000 scheduler.first_cycle_steps=5000`. The header of the file
   lists the settings you are most likely to change. The model block must stay as it is. A checkpoint with its
   optimizer state is about 4.4 GB and is assembled in memory before it is written, so train on a node with at
   least 16 GB of RAM; on a login node with an 8 GB per-user limit the run stalls at the first checkpoint.

3. Use the result. Checkpoints land in `checkpoints/`. Strip one to a weights file and load it as before:

   ```bash
   tace-export-train -m checkpoints/last.ckpt -o my_model.pt --dtype float32
   ```

## Accuracy and limits

These numbers are for `DefectNet-v1.0.pt` and are reported in the manuscript.

- 10,573 held-out DFT frames: energy RMSE 5.0 meV/atom (3.9 meV/atom for the 9,304 defect-supercell frames, 2.3
  meV/atom for PBEsol), force RMSE 13.9 meV/Å. Without the QEq head (charge embedding only) the energy RMSE on the
  same frames is 5.6 meV/atom.
- 172 PBEsol charge states in 15 semiconductors (216–289 atoms) relaxed from rattled starts, each side with
  chemical potentials from its own energies of the competing phases: formation-energy RMSE 0.34 eV; transition-level
  RMSE 0.22 eV over 134 levels and 0.21 eV over the 58 inside the gap.
- Vacancy migration barriers by climbing-image NEB for 17 hops in charge states from −2 to +2: RMSE 0.078 eV against
  DFT NEB, 0.043 eV over the 14 cation-vacancy hops.
- The PBE and hybrid heads give formation energies off by about 1 eV (root-mean-square) even on geometries relaxed
  with the PBEsol head. Use the PBEsol head for defect energetics.
- Almost all benchmark cells come from semiconductors that were also in training. Other chemistries fall back on
  the foundation model and have not been tested; fine-tune before you rely on them.
- In some charge states a DefectNet minimum can differ from the DFT one (for example an off-site substitutional).
  Check structures that matter with a DFT relaxation started from the DefectNet geometry.

## Repository

```
defectnet/          the package: calculator, qeq, relax, formation, neb, dataset, train
examples/           three runnable scripts, tutorial.ipynb and its CdTe data, paper/ notebooks
finetune/           fine-tune template (finetune.yaml)
patches/            the TACE fix the weights need
tests/              pytest; the model tests run when DEFECTNET_MODEL points at the weights
```

## Citation

The DefectNet paper is in preparation. Please cite the TACE architecture and the foundation model it starts from:

- Z. Xu, W. Xie and P. Hu, Spectral/Spatial Tensor Atomic Cluster Expansion with Universal Embeddings in Cartesian
  Space, [arXiv:2509.14961](https://doi.org/10.48550/arXiv.2509.14961) (2025).
- Z. Xu, W. Xie and P. Hu, Edge Cluster Expansion with Radial Rotary Attention for Interatomic Potentials,
  [arXiv:2607.10664](https://doi.org/10.48550/arXiv.2607.10664) (2026).
- Z. Xu, TACE foundation checkpoints (xvzemin/tace-foundations), Hugging Face,
  [doi:10.57967/hf/7458](https://doi.org/10.57967/hf/7458) (2026).

TACE is MIT-licensed; the patch in `patches/` modifies its code under that licence.
