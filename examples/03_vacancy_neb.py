"""Migration barrier of the Cd vacancy in CdTe for two charge states, climbing-image NEB with DefectNet.

python 03_vacancy_neb.py DefectNet-v1.0.pt
"""
import sys

from ase.build import bulk
from ase.io import write

import defectnet as dn

calc = dn.load(sys.argv[1], functional="PBEsol")
host = bulk("CdTe", "zincblende", a=6.48, cubic=True).repeat(3)
for q in (0, -2):
    r = dn.vacancy_hop(host, site=0, calc=calc, q=q)
    print("V_Cd q = %+d  hop %.2f A  barrier %.3f eV  converged %s" % (q, r["hop_length"], r["Em_fwd"], r["converged"]))
    write("neb_VCd_q%+d.extxyz" % q, r["images"])
