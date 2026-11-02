"""Fixed-cell relaxation of a (charged) defect supercell, the protocol used in the DefectNet benchmark."""
import numpy as np
from ase.optimize import FIRE


def rattle(atoms, seed=0):
    """Displace every atom by a Gaussian of standard deviation 10 % of the shortest interatomic distance
    (the ShakeNBreak default; 0.25 A if that falls outside 0.02-0.4 A). Breaks the symmetry of an ideal start."""
    a = atoms.copy()
    d = a.get_all_distances(mic=True)
    sd = 0.1 * float(d[d > 1e-6].min()) if len(a) > 1 else 0.25
    if not 0.02 <= sd <= 0.4:
        sd = 0.25
    a.rattle(stdev=sd, seed=seed)
    return a


def relax(atoms, calc, q=0, fmax=0.02, steps=1000, shake=True, seed=0, trajectory=None):
    """Relax the atoms (cell fixed) with FIRE at total charge q. Returns the relaxed copy; its info holds
    "energy" (eV), "converged" and "steps"."""
    a = rattle(atoms, seed) if shake else atoms.copy()
    a.info["total_charge"] = float(q)
    a.calc = calc
    opt = FIRE(a, logfile=None, trajectory=trajectory)
    ok = opt.run(fmax=fmax, steps=steps)
    a.info.update(energy=float(a.get_potential_energy()), converged=bool(ok), steps=opt.nsteps,
                  fmax=float(np.linalg.norm(a.get_forces(), axis=1).max()))
    return a
