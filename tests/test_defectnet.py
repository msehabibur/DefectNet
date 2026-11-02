"""pytest tests. The model tests run only when DEFECTNET_MODEL points at the weights file."""
import os

import numpy as np
import pytest
from ase.build import bulk

import defectnet as dn


def levels(ef0, gap):
    return [(round(e, 9), q1, q2) for e, q1, q2 in dn.transition_levels(ef0, gap)]


def test_transition_levels():
    # E_f(q, E_F) = ef0[q] + q E_F: +1 below 0.5 eV, 0 up to 1.2 eV, -1 above
    ef0 = {1: 1.0, 0: 1.5, -1: 2.7}
    assert levels(ef0, 2.0) == [(0.5, 1, 0), (1.2, 0, -1)]
    assert levels(ef0, 1.0) == [(0.5, 1, 0)]
    assert dn.lowest(ef0, 1.5) == (-1, pytest.approx(1.2))


def test_negative_u():
    # (+1/0) lies above (0/-1), so q = 0 is never stable and the level is (+1/-1) at 0.75 eV
    ef0 = {1: 1.0, 0: 2.0, -1: 2.5}
    assert levels(ef0, 2.0) == [(0.75, 1, -1)]


def test_formation_energy():
    e = dn.formation_energy(-100.0, -103.0, {"Cd": -1}, {"Cd": -1.0}, q=-2, vbm=2.0, e_fermi=0.5)
    assert e == pytest.approx(-100.0 + 103.0 - 1.0 - 2 * 2.5)


MODEL = os.environ.get("DEFECTNET_MODEL")


@pytest.mark.skipif(not MODEL, reason="set DEFECTNET_MODEL to the weights file")
def test_model_charge_and_relax():
    calc = dn.load(MODEL, functional="PBEsol", device="cpu")
    cell = bulk("CdTe", "zincblende", a=6.48, cubic=True)
    e = {}
    for q in (0, 1):
        a = dn.charged(cell, q)
        a.calc = calc
        e[q] = a.get_potential_energy()
        assert np.isfinite(e[q])
    assert abs(e[1] - e[0]) > 1e-3                   # the charge is read (ignored, the two would be identical)
    vac = cell.copy()
    del vac[0]
    r = dn.relax(vac, calc, q=-1, steps=300)
    assert r.info["converged"] and r.info["fmax"] < 0.02
