"""Load DefectNet as an ASE calculator.

DefectNet has one output head per level of theory and reads the total charge of the cell from
atoms.info["total_charge"] (q = +1 means one electron removed). Without that key the charge is 0.
"""
import torch

# head index of each level of theory, in the order the model was trained with
FUNCTIONALS = {
    "PBEsol": 0,
    "PBE": 1,
    "PBE+U": 2,
    "HSE06": 3,           # HSE06 on the PBE pseudopotential set
    "HSE06(PBEsol)": 4,   # HSE06 with GGA = PS
    "HSE06+SOC": 5,       # HSE06(PBEsol) with spin-orbit coupling
}


def _device(device):
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def load(model, functional="PBEsol", device=None):
    """Return an ASE calculator for the weights file `model` (.pt or Lightning .ckpt).

    functional: a key of FUNCTIONALS or a head index. device: "cuda", "xpu", "cpu" or None (best available).
    """
    from tace.interface.ase import GeneralTACEAseCalc
    from . import qeq  # noqa: F401  (periodic QEq electrostatics, needed by the QEq head of the weights)
    head = FUNCTIONALS[functional] if isinstance(functional, str) else int(functional)
    device = _device(device)
    if device == "xpu":                     # TACE 0.2.1 has no entry for Intel GPUs
        from tace.utils._global import DEVICE
        DEVICE["xpu"] = DEVICE[torch.device("xpu")] = torch.device("xpu")
    return GeneralTACEAseCalc(model=str(model), device=device, fidelity_idx=head,
                              keys={"total_charge": "total_charge"})


def charged(atoms, q):
    """Copy of `atoms` with total charge q, ready for a DefectNet calculator."""
    a = atoms.copy()
    a.info["total_charge"] = float(q)
    return a
