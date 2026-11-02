"""Energy of a CdTe supercell at each level of theory, then a charged Cd vacancy relaxed with DefectNet.

python 01_quickstart.py DefectNet-v1.0.pt
"""
import sys

from ase.build import bulk

import defectnet as dn

model = sys.argv[1]
cell = bulk("CdTe", "zincblende", a=6.48, cubic=True).repeat(3)          # 216 atoms, the training cell size
for name in dn.FUNCTIONALS:
    a = dn.charged(cell, 0)
    a.calc = dn.load(model, functional=name)
    print("%-14s E = %10.4f eV  (%.4f eV/atom)" % (name, a.get_potential_energy(), a.get_potential_energy() / len(a)))

calc = dn.load(model, functional="PBEsol")
vac = cell.copy()
del vac[0]                                                               # V_Cd
for q in (0, -1, -2):
    r = dn.relax(vac, calc, q=q)
    print("V_Cd q = %+d   E = %.4f eV   %d FIRE steps, converged %s" % (q, r.info["energy"], r.info["steps"], r.info["converged"]))
