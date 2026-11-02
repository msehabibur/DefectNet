# Paper notebooks

One notebook per figure panel of the DefectNet paper. Each one loads the DFT data from `data/`, redraws the panel,
and reruns DefectNet on one example, printing its numbers next to the ones in the paper.

| notebook | panel | what it computes with DefectNet | time on one A100 |
|---|---|---|---|
| `fig6a_formation_energy.ipynb` | Fig. 6a | relaxations of V<sub>Cu</sub> in CuGaSeTe in five charge states, with their FIRE trajectories (all 172 cells with `RUN_ALL = True`) | 6 min |
| `fig7_fermi_level.ipynb` | Fig. 7 | the 14 charge states of CdSe<sub>0.06</sub>Te<sub>0.94</sub>, their formation energies on the HSE06 band edges and the self-consistent Fermi level | 8 min |
| `fig8a_configuration_coordinate.ipynb` | Fig. 8a | configuration coordinate diagrams and capture barriers of Bi<sub>Cd</sub> + Bi<sub>Se</sub> in CdSe<sub>0.06</sub>Te<sub>0.94</sub>, and its local structure against DFT | 10 min |
| `fig8b_vacancy_neb.ipynb` | Fig. 8b | climbing-image NEB of V<sub>Cd</sub><sup>2-</sup>, V<sub>Ag</sub><sup>-</sup> and V<sub>Cu</sub><sup>-</sup> | 36 min |
| `fig8d_phonons.ipynb` | Fig. 8d | Γ-point phonons and vibrational entropy of the same complex in five charge states | over 60 min |

Run them from this folder. The first cell installs what is missing and downloads `DefectNet-v1.0.pt` from the v1.0
release (or uses a copy in this folder or in `examples/`).

## Data

The tables are Apache Arrow (Feather v2, uncompressed) and open with `pyarrow.feather.read_table(...)` or
`pandas.read_feather(...)`. All energies are in eV, positions and cells in Å. The DFT is VASP with PBEsol at a fixed
cell; no finite-size correction is applied anywhere.

`pristine_pbesol.arrow`, 15 rows, one per semiconductor:

| column | content |
|---|---|
| `path` | name of the DFT calculation (also the seed of its rattle, see the notebooks) |
| `host`, `natoms` | semiconductor and number of atoms |
| `E_dft`, `vbm`, `gap` | DFT total energy, valence-band maximum and band gap (PBEsol) |
| `E_defectnet_paper` | DefectNet energy after the benchmark relaxation, as in the paper |
| `start_symbols`, `start_positions`, `start_cell` | DFT final geometry, the start of the DefectNet relaxation |

`fig6a_cells.arrow`, 172 rows, one per charge state of Fig. 6a, and `fig7_cells.arrow`, 52 rows, one per charge state
of Fig. 7, with the same columns:

| column | content |
|---|---|
| `path`, `pristine` | name of the DFT calculation and of its pristine cell |
| `host`, `defect`, `charge`, `q` | semiconductor, defect, charge-state label and total charge (q = +1: one electron removed) |
| `natoms`, `E_dft` | number of atoms and DFT total energy |
| `vbm`, `gap` | VBM and band gap used for this cell's formation energy in the paper (PBEsol, pristine cell) |
| `change_elements`, `change_counts` | atoms added (+1) and removed (-1) relative to the pristine cell |
| `vertex`, `mu_elements` | chemical-potential limit of the benchmark and its elements |
| `mu_values`, `mu_defectnet` | chemical potentials at that limit (eV/atom) from DFT and from DefectNet energies of the competing phases |
| `Ef0_dft`, `Ef0_defectnet_paper` | formation energy at E<sub>F</sub> = VBM from DFT and from DefectNet (paper), each with its own chemical potentials |
| `E_defectnet_paper` | DefectNet energy after the benchmark relaxation |
| `start_symbols`, `start_positions`, `start_cell` | DFT input geometry, the start of both relaxations |
| `dft_positions` | DFT final geometry (same atoms and cell) |

`fig7_compounds.arrow`, 12 rows, every chemical-potential limit of the three semiconductors of Fig. 7:

| column | content |
|---|---|
| `host`, `vertex`, `label`, `facet` | semiconductor, index and name of the limit, phases in equilibrium there |
| `mu_elements`, `mu_values` | chemical potentials (eV/atom), dopants at the caps of the same limit |
| `mu_defectnet` | the same from DefectNet energies, at the matching limit of the DefectNet phase diagram |
| `vbm_hse`, `gap_hse` | HSE06 VBM and band gap of the semiconductor (VASP, AEXX 0.25, on PBEsol) |
| `shown` | the limit drawn in the paper |
| `volume_A3` | volume of the pristine supercell, for the site densities of the self-consistent Fermi level |
