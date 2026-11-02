"""Vacancy migration barrier with climbing-image NEB, the protocol of the DefectNet paper (Fig. 8)."""
import numpy as np
from ase.optimize import FIRE, LBFGS
try:
    from ase.mep import NEB
except ImportError:                         # ase < 3.23
    from ase.neb import NEB


def _mic(a, d):
    f = np.linalg.solve(a.cell.T, np.atleast_2d(d).T).T
    return (f - np.round(f)) @ a.cell.array


def vacancy_hop(pristine, site, calc, q=0, neighbour=None, nimages=7, fmax=0.03):
    """Barrier for the hop of a vacancy at `site` (index in `pristine`) to a like neighbour.

    Initial state: atom `site` removed. Final state: the same cell with the neighbour moved onto `site`.
    `neighbour` defaults to the nearest atom of the same element. Both end points are relaxed (LBFGS, 0.01 eV/A),
    the path is interpolated with IDPP, relaxed as a plain band to 0.1 eV/A and then with a climbing image to `fmax`.
    Returns a dict with the image energies E (eV), the forward and backward barriers Em_fwd, Em_bwd and the images.
    """
    sym = np.array(pristine.get_chemical_symbols())
    d = pristine.get_all_distances(mic=True)[site]
    if neighbour is None:
        like = [k for k in range(len(pristine)) if k != site and sym[k] == sym[site]]
        neighbour = min(like, key=lambda k: d[k])
    keep = [k for k in range(len(pristine)) if k != site]
    ini = pristine[keep]
    fin = ini.copy()
    j = keep.index(neighbour)
    p = pristine.positions
    fin.positions[j] = p[neighbour] + _mic(pristine, p[site] - p[neighbour])[0]
    ends = []
    for x in (ini, fin):
        x.info["total_charge"] = float(q)
        x.calc = calc
        LBFGS(x, logfile=None).run(fmax=0.01, steps=2000)
        ends.append(x)
    images = [ends[0]] + [ends[0].copy() for _ in range(nimages)] + [ends[1]]
    for im in images:
        im.info["total_charge"] = float(q)
        im.calc = calc
    band = NEB(images, climb=False, allow_shared_calculator=True)
    band.interpolate(method="idpp", mic=True)
    FIRE(band, logfile=None).run(fmax=0.1, steps=600)
    band.climb = True
    ok = FIRE(band, logfile=None).run(fmax=fmax, steps=1500)
    E = [float(im.get_potential_energy()) for im in images]
    return dict(E=E, Em_fwd=max(E) - E[0], Em_bwd=max(E) - E[-1], converged=bool(ok),
                hop_length=float(d[neighbour]), neighbour=int(neighbour), images=images)
