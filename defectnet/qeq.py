"""Periodic charge-equilibration (QEq) electrostatics for DefectNet cells, imported by defectnet.load.

TACE 0.2.1 graphs carry the lattice but no pbc flag, and GraphLongRangeElectrostatics then treats every crystal as an
open cluster. With this patch a graph without a pbc flag is periodic in all three directions when its lattice is
non-singular (every DefectNet cell). It also replaces GTOBasis._prepare_k_moduli, which fills the output of a sqrt in
place and so breaks the force and stress backward pass ("modified by an inplace operation"), with an out-of-place
version that is safe at k = 0. DefectNet was trained and evaluated with exactly this patch.
"""
import torch
from tace.models.scf.electrostatics import GraphLongRangeElectrostatics
from tace.models.scf.longrange.gto_utils import GTOBasis

_geometry = GraphLongRangeElectrostatics._geometry


def _periodic_geometry(self, data, graph=None):
    if data.get("pbc") is not None:
        return _geometry(self, data, graph)
    cell = data["lattice"].reshape(-1, 3, 3)
    d = {"positions": data["positions"], "batch": data["batch"], "lattice": cell,
         "pbc": (torch.linalg.det(cell).abs() > 1e-6)[:, None].expand(-1, 3)}
    if "ptr" in data:
        d["ptr"] = data["ptr"]
    return _geometry(self, d, graph)


GraphLongRangeElectrostatics._geometry = _periodic_geometry


def _prepare_k_moduli(self, k_norm2, k0_mask):
    k0 = k0_mask > 0.0
    return torch.sqrt(torch.where(k0, torch.ones_like(k_norm2), k_norm2.clamp_min(0.0))).masked_fill(k0, 0.0)


GTOBasis._prepare_k_moduli = _prepare_k_moduli
