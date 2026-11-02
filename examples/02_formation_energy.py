"""Formation energy of V_Cd in CdTe against the Fermi level, every energy from DefectNet (PBEsol head).

Chemical potentials (Te-rich limit): mu_Te = E(Te) per atom, mu_Cd = E(CdTe) per formula unit - mu_Te.
Reference cells are relaxed with their lattice free (FrechetCellFilter); the defect cell keeps the relaxed CdTe lattice.
DefectNet does not predict band edges, so the valence-band maximum and the gap come from your own DFT run of the
pristine supercell at the same level of theory:

python 02_formation_energy.py DefectNet-v1.0.pt --vbm <VBM in eV> --gap <gap in eV>
"""
import argparse

from ase.build import bulk
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from ase.spacegroup import crystal

import defectnet as dn

p = argparse.ArgumentParser()
p.add_argument("model")
p.add_argument("--vbm", type=float, required=True, help="VBM of the pristine 216-atom CdTe cell (eV)")
p.add_argument("--gap", type=float, required=True, help="band gap (eV)")
a = p.parse_args()
calc = dn.load(a.model, functional="PBEsol")


def relaxed(atoms):
    atoms = dn.charged(atoms, 0)
    atoms.calc = calc
    FIRE(FrechetCellFilter(atoms), logfile=None).run(fmax=0.01, steps=500)
    return atoms


cdte = relaxed(bulk("CdTe", "zincblende", a=6.48))
te = relaxed(crystal("Te", [(0.2636, 0, 1 / 3)], spacegroup=152, cellpar=[4.457, 4.457, 5.929, 90, 90, 120]))
mu = {"Te": te.get_potential_energy() / len(te)}
mu["Cd"] = cdte.get_potential_energy() / 1 - mu["Te"]                   # the primitive cell is one formula unit
print("CdTe a = %.3f A   mu_Cd = %.4f   mu_Te = %.4f eV/atom" % (cdte.cell.lengths()[0] * 2 ** 0.5, mu["Cd"], mu["Te"]))

host = bulk("CdTe", "zincblende", a=cdte.cell.lengths()[0] * 2 ** 0.5, cubic=True).repeat(3)
e_bulk = dn.relax(host, calc, q=0, shake=False).info["energy"]
vac = host.copy()
del vac[0]
ef0 = {}
for q in (2, 1, 0, -1, -2):
    e = dn.relax(vac, calc, q=q).info["energy"]
    ef0[q] = dn.formation_energy(e, e_bulk, {"Cd": -1}, mu, q, a.vbm)
    print("V_Cd q = %+d   E_f(E_F = VBM) = %.3f eV" % (q, ef0[q]))
for e, q1, q2 in dn.transition_levels(ef0, a.gap):
    print("level (%+d/%+d) at E_F = %.3f eV above the VBM" % (q1, q2, e))
