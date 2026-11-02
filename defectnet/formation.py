"""Defect formation energies and charge-transition levels.

E_f(q, E_F) = E_def(q) - E_bulk - sum_i n_i mu_i + q (E_VBM + E_F) + E_corr
n_i = +1 for an atom added to the supercell and -1 for an atom removed; E_F is measured from the VBM.
"""
import numpy as np


def formation_energy(e_defect, e_bulk, change, mu, q, vbm, e_fermi=0.0, correction=0.0):
    """change: {element: n_i}, e.g. {"Cd": -1} for V_Cd or {"Cu": +1, "Cd": -1} for Cu_Cd.
    mu: {element: chemical potential in eV/atom}, on the same energy scale as e_defect and e_bulk.
    vbm: valence-band maximum of the pristine cell (eV) at the same level of theory."""
    return e_defect - e_bulk - sum(n * mu[el] for el, n in change.items()) + q * (vbm + e_fermi) + correction


def lowest(ef0, e_fermi):
    """ef0: {q: E_f at E_F = 0}. Returns (q, E_f) of the most stable charge state at e_fermi."""
    q = min(ef0, key=lambda k: ef0[k] + k * e_fermi)
    return q, ef0[q] + q * e_fermi


def transition_levels(ef0, gap):
    """Fermi levels (eV above the VBM) inside [0, gap] where the most stable charge state changes.
    Returns a list of (E_F, q_before, q_after), q_before being the charge state below that level."""
    out = []
    e, (q, _) = 0.0, lowest(ef0, 0.0)
    while True:
        # next crossing of the current state q with any lower charge state
        nxt = [((ef0[k] - ef0[q]) / (q - k), k) for k in ef0 if k < q]
        nxt = [(x, k) for x, k in nxt if x > e + 1e-12]
        if not nxt:
            return out
        x, k = min(nxt, key=lambda t: (t[0], t[1]))
        if x >= gap:
            return out
        k = lowest(ef0, x + 1e-9)[0]
        out.append((float(x), q, k))
        e, q = x, k


def fermi_grid(ef0, gap, n=201):
    """E_F grid over the gap and the formation-energy envelope min_q E_f(q, E_F) on it."""
    x = np.linspace(0.0, gap, n)
    return x, np.min([[ef0[q] + q * e for q in ef0] for e in x], axis=1)
