"""Turn your own DFT runs into DefectNet training frames (extended XYZ).

Every ionic step of each run becomes one frame with energy, forces, stress (when the run has it), the head
index of its level of theory (fidelity_idx) and its total charge. ASE reads the run, so anything ase.io.read
understands works: VASP vasprun.xml or OUTCAR, Quantum ESPRESSO pw.x output, CP2K, FHI-aims, ...

    defectnet-data --functional PBEsol --charge -1 -o train.xyz V_Cd_q-1/vasprun.xml
    defectnet-data --functional PBE --charge 0 --every 5 -o train.xyz relax/pw.out     (appends)
"""
import argparse

from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator

from .calculator import FUNCTIONALS


def frames(path, functional, charge, every=1, last=False):
    head = FUNCTIONALS[functional] if functional in FUNCTIONALS else int(functional)
    steps = read(path, index=":")
    steps = steps[-1:] if last else steps[::every]
    out = []
    for a in steps:
        res = dict(energy=a.get_potential_energy(), forces=a.get_forces())
        try:
            res["stress"] = a.get_stress()
        except Exception:                   # not every code writes a stress at every step
            pass
        b = a.copy()
        b.calc = SinglePointCalculator(b, **res)
        b.info.update(fidelity_idx=head, total_charge=float(charge), source=str(path))
        out.append(b)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="DFT output files, all at the same level of theory and charge")
    p.add_argument("--functional", required=True, help="one of %s, or a head index" % ", ".join(FUNCTIONALS))
    p.add_argument("--charge", type=float, default=0.0, help="total charge q of the cell (+1 = one electron removed)")
    p.add_argument("--every", type=int, default=1, help="keep every n-th ionic step")
    p.add_argument("--last", action="store_true", help="keep only the final step of each run")
    p.add_argument("-o", "--output", required=True, help="extended XYZ file; frames are appended")
    a = p.parse_args()
    n = 0
    for r in a.runs:
        f = frames(r, a.functional, a.charge, a.every, a.last)
        write(a.output, f, format="extxyz", append=True)
        n += len(f)
    print("wrote %d frames to %s" % (n, a.output))


if __name__ == "__main__":
    main()
