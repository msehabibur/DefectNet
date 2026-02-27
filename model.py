"""
model.py

Architecture
============
Input:  periodic crystal structure
        (atom types, positions, lattice, neighbour graph, triplet indices)

1.  Atom embedding              Z  -->  h_i   (learnable, indexed by atomic number)
2.  Gaussian distance expansion  d  -->  e_ij  (fixed radial basis)
3.  Smooth cosine cutoff         d  -->  w_ij  (ensures force continuity)
4.  N × interaction blocks, each containing:
      a)  2-body DefectNet convolution   (pairwise distance message-passing)
      b)  3-body angular convolution (triplet angle message-passing)
5.  Per-atom energy head          h_i -->  ε_i
6.  Total energy                  E = Σ_i ε_i   (sum pooling – extensive)
7.  Forces                        F = -dE/dr     (autograd)
8.  Stress                        σ = (1/V) dE/dε (strain-derivative method)

Units (when used with VASP data)
================================
Energy  :  eV
Forces  :  eV / Å
Stress  :  eV / ų   (multiply by 1602.1766 to get kBar)
"""

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
#  Helper modules
# ---------------------------------------------------------------------------

class GaussianSmearing(nn.Module):
    """Expand scalar distances into a fixed Gaussian basis."""

    def __init__(self, d_min: float = 0.0, d_max: float = 5.0,
                 num_gaussians: int = 80, variance: float = 0.2):
        super().__init__()
        centers = torch.linspace(d_min, d_max, num_gaussians)
        self.register_buffer("centers", centers)
        self.variance = variance

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """distances: (E,)  →  (E, num_gaussians)"""
        return torch.exp(
            -(distances.unsqueeze(-1) - self.centers) ** 2
            / (self.variance ** 2)
        )


class AngularBasis(nn.Module):
    """
    Expand cos(θ) into a fixed Gaussian basis over [-1, +1].

    More numerically stable than Fourier expansion of the raw angle
    because it avoids acos near ±1.
    """

    def __init__(self, num_basis: int = 16, variance: float = 0.15):
        super().__init__()
        centers = torch.linspace(-1.0, 1.0, num_basis)
        self.register_buffer("centers", centers)
        self.variance = variance

    def forward(self, cos_angle: torch.Tensor) -> torch.Tensor:
        """cos_angle: (T,)  →  (T, num_basis)"""
        return torch.exp(
            -(cos_angle.unsqueeze(-1) - self.centers) ** 2
            / (self.variance ** 2)
        )


class CosineCutoff(nn.Module):
    """Smooth cosine cutoff  [1 at d=0, 0 at d=cutoff]."""

    def __init__(self, cutoff: float):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """distances: (E,)  →  (E,)"""
        return 0.5 * (torch.cos(distances * math.pi / self.cutoff) + 1.0)


# ---------------------------------------------------------------------------
#  2-body: DefectNet convolution layer
# ---------------------------------------------------------------------------

class DefectNetConv(nn.Module):
    """
    DefectNet convolution  (Xie & Grossman, PRL 2018).

    z_i' = softplus( z_i + BN( Σ_j  σ(W[z_i, z_j, e_ij]) ⊙ sp(W[z_i, z_j, e_ij]) ) )

    Edge convention:  src = neighbour (sender),  dst = centre (receiver).
    """

    def __init__(self, atom_fea_len: int, edge_fea_len: int):
        super().__init__()
        self.fc = nn.Linear(2 * atom_fea_len + edge_fea_len,
                            2 * atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * atom_fea_len)
        self.bn2 = nn.BatchNorm1d(atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_fea, edge_fea, edge_index, cutoff_w=None):
        """
        Parameters
        ----------
        atom_fea   : (N, F)   atom features
        edge_fea   : (E, G)   edge features (Gaussian expansion)
        edge_index : (2, E)   [src=neighbour, dst=centre]
        cutoff_w   : (E,)     optional smooth cutoff weights

        Returns
        -------
        atom_fea   : (N, F)   updated atom features
        """
        src, dst = edge_index

        # Build message input  [centre, neighbour, edge]
        z_centre = atom_fea[dst]
        z_nbr    = atom_fea[src]
        total = torch.cat([z_centre, z_nbr, edge_fea], dim=1)   # (E, 2F+G)

        total = self.bn1(self.fc(total))                         # (E, 2F)

        gate, core = total.chunk(2, dim=1)                       # each (E, F)
        gate = self.sigmoid(gate)
        core = self.softplus1(core)

        messages = gate * core                                   # (E, F)

        if cutoff_w is not None:
            messages = messages * cutoff_w.unsqueeze(-1)

        # Aggregate messages to centre (dst) atoms
        aggr = torch.zeros_like(atom_fea)
        aggr.index_add_(0, dst, messages)

        aggr = self.bn2(aggr)
        return self.softplus2(atom_fea + aggr)                   # residual


# ---------------------------------------------------------------------------
#  3-body: Angular convolution layer
# ---------------------------------------------------------------------------

class ThreeBodyConv(nn.Module):
    """
    Angular (3-body) message-passing layer.

    For every centre atom *i* and each pair of neighbours (j, k),
    compute cos(θ_{j,i,k}) from the displacement vectors r_ij and r_ik,
    expand into an angular Gaussian basis, combine with both radial edge
    features, and aggregate a gated message back to atom *i*.

    This captures bond-angle information that purely 2-body DefectNet misses.
    """

    def __init__(
        self,
        atom_fea_len: int,
        edge_fea_len: int,
        num_angular_basis: int = 16,
    ):
        super().__init__()
        self.angular_expansion = AngularBasis(
            num_basis=num_angular_basis, variance=0.15,
        )

        # Input: [centre_atom_fea, edge_fea_1, edge_fea_2, angular_basis]
        in_dim = atom_fea_len + 2 * edge_fea_len + num_angular_basis
        self.fc = nn.Linear(in_dim, 2 * atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * atom_fea_len)
        self.bn2 = nn.BatchNorm1d(atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(
        self,
        atom_fea,       # (N, F)
        edge_fea,       # (E, G)   Gaussian radial features
        r_ij,           # (E, 3)   displacement vectors
        dist,           # (E,)     interatomic distances
        edge_index,     # (2, E)   [src, dst]
        triplet_idx,    # (2, T)   [primary_edge, secondary_edge]
        cutoff_w=None,  # (E,)     optional smooth cutoff weights
    ):
        """
        Returns
        -------
        atom_fea : (N, F)  updated atom features
        """
        if triplet_idx.shape[1] == 0:
            # no triplets (degenerate graph) → identity
            return atom_fea

        e1, e2 = triplet_idx                                    # each (T,)
        dst = edge_index[1]

        # ---- cos(angle) between displacement pairs ----
        v1 = r_ij[e1]                                           # (T, 3)
        v2 = r_ij[e2]                                           # (T, 3)
        d1 = dist[e1].clamp(min=1e-8)                           # (T,)
        d2 = dist[e2].clamp(min=1e-8)                           # (T,)
        cos_angle = (v1 * v2).sum(dim=-1) / (d1 * d2)           # (T,)
        cos_angle = cos_angle.clamp(-1.0, 1.0)

        # ---- angular features ----
        ang_fea = self.angular_expansion(cos_angle)              # (T, A)

        # ---- centre atom features for each triplet ----
        centre_atoms = dst[e1]                                   # (T,)
        centre_fea = atom_fea[centre_atoms]                      # (T, F)

        # ---- combine: [centre, edge1, edge2, angular] ----
        triplet_fea = torch.cat([
            centre_fea, edge_fea[e1], edge_fea[e2], ang_fea,
        ], dim=1)                                                # (T, F+2G+A)

        triplet_fea = self.bn1(self.fc(triplet_fea))             # (T, 2F)

        # ---- gating (same pattern as DefectNet 2-body) ----
        gate, core = triplet_fea.chunk(2, dim=1)                 # each (T, F)
        gate = self.sigmoid(gate)
        core = self.softplus1(core)
        messages = gate * core                                   # (T, F)

        # ---- optional cutoff weight (product of both radial cutoffs) ----
        if cutoff_w is not None:
            w_triplet = cutoff_w[e1] * cutoff_w[e2]              # (T,)
            messages = messages * w_triplet.unsqueeze(-1)

        # ---- aggregate to centre atoms ----
        aggr = torch.zeros_like(atom_fea)
        aggr.index_add_(0, centre_atoms, messages)

        aggr = self.bn2(aggr)
        return self.softplus2(atom_fea + aggr)                   # residual


# ---------------------------------------------------------------------------
#  Full force-field model
# ---------------------------------------------------------------------------

class DefectNetForceField(nn.Module):
    """
    DefectNet-style graph neural-network force field with 2-body + 3-body terms.

    Each interaction block contains:
      1.  2-body DefectNet conv  (pairwise radial messages)
      2.  3-body angular conv  (triplet angular messages)
    Both update atom features with residual connections.

    Predicts total energy E, per-atom forces F = -dE/dr, and
    stress tensor σ = (1/V) dE/dε via the strain-derivative method.
    """

    def __init__(
        self,
        num_species: int       = 119,       # 0 = pad, 1..118 = elements
        atom_fea_len: int      = 64,        # atom embedding / hidden dim
        num_conv: int          = 4,         # number of interaction blocks
        num_gaussians: int     = 80,        # radial Gaussian basis functions
        num_angular_basis: int = 16,        # angular Gaussian basis functions
        cutoff: float          = 5.0,       # neighbour cutoff  [Angstrom]
        energy_hidden: tuple   = (128, 64), # FC layers before per-atom energy
        num_global_features: int = 2,       # charge + level_of_theory
    ):
        super().__init__()
        self.cutoff = cutoff
        self.num_global_features = num_global_features

        # --- atom embedding ---
        self.atom_embedding = nn.Embedding(num_species, atom_fea_len)

        # --- global feature projection (charge, theory → atom_fea_len) ---
        if num_global_features > 0:
            self.global_projection = nn.Sequential(
                nn.Linear(num_global_features, atom_fea_len),
                nn.Softplus(),
            )

        # --- radial features ---
        self.distance_expansion = GaussianSmearing(
            d_min=0.0, d_max=cutoff, num_gaussians=num_gaussians,
        )
        self.cutoff_fn = CosineCutoff(cutoff)

        # --- interaction blocks (2-body + 3-body) ---
        self.two_body_convs = nn.ModuleList([
            DefectNetConv(atom_fea_len, num_gaussians)
            for _ in range(num_conv)
        ])
        self.three_body_convs = nn.ModuleList([
            ThreeBodyConv(atom_fea_len, num_gaussians, num_angular_basis)
            for _ in range(num_conv)
        ])

        # --- per-atom energy head ---
        layers = []
        in_dim = atom_fea_len
        for h in energy_hidden:
            layers += [nn.Linear(in_dim, h), nn.Softplus()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.energy_head = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    def forward(self, data, compute_forces=True, compute_stress=True):
        """
        Parameters
        ----------
        data : dict
            atom_types   : (N,)     long    – atomic numbers
            pos          : (N, 3)   float   – Cartesian positions
            edge_index   : (2, E)   long    – [src, dst]
            edge_offset  : (E, 3)   float   – Cartesian PBC offsets
            triplet_idx  : (2, T)   long    – [primary_edge, secondary_edge]
            batch        : (N,)     long    – maps atoms → structures
            num_atoms    : (B,)     long
            volume       : (B,)     float
        compute_forces  : bool
        compute_stress  : bool

        Returns
        -------
        dict  with keys  'energy' (B,), 'forces' (N,3), 'stress' (B,3,3)
        """
        atom_types  = data["atom_types"]
        pos         = data["pos"]
        edge_index  = data["edge_index"]
        edge_offset = data["edge_offset"]
        triplet_idx = data["triplet_idx"]
        batch       = data["batch"]
        num_atoms   = data["num_atoms"]
        volume      = data["volume"]
        device      = pos.device
        batch_size  = num_atoms.shape[0]

        # ---- enable position gradients for forces ----
        if compute_forces and not pos.requires_grad:
            pos = pos.requires_grad_(True)

        # ---- strain tensor for stress  (evaluated at ε = 0) ----
        strain = None
        if compute_stress:
            strain = torch.zeros(
                batch_size, 3, 3, device=device, dtype=pos.dtype,
                requires_grad=True,
            )
            # deformation matrix per structure:  D_k = I + ε_k
            deformation = (
                torch.eye(3, device=device).unsqueeze(0) + strain
            )                                                    # (B, 3, 3)

            # broadcast to per-atom / per-edge
            per_atom_D = deformation[batch]                      # (N, 3, 3)
            pos_d = torch.bmm(
                pos.unsqueeze(1), per_atom_D.transpose(1, 2),
            ).squeeze(1)                                         # (N, 3)

            src = edge_index[0]
            per_edge_D = deformation[batch[src]]                 # (E, 3, 3)
            offset_d = torch.bmm(
                edge_offset.unsqueeze(1), per_edge_D.transpose(1, 2),
            ).squeeze(1)                                         # (E, 3)
        else:
            pos_d    = pos
            offset_d = edge_offset

        # ---- displacement vectors & distances ----
        src, dst = edge_index
        r_ij = pos_d[src] + offset_d - pos_d[dst]               # (E, 3)
        dist = torch.norm(r_ij, dim=-1)                          # (E,)

        # ---- edge features ----
        edge_fea = self.distance_expansion(dist)                 # (E, G)
        cutoff_w = self.cutoff_fn(dist)                          # (E,)

        # ---- atom features ----
        atom_fea = self.atom_embedding(atom_types)               # (N, F)

        # ---- condition on global features (charge, theory) ----
        if self.num_global_features > 0 and "global_features" in data:
            global_fea = data["global_features"]                 # (B, G)
            global_proj = self.global_projection(global_fea)     # (B, F)
            atom_fea = atom_fea + global_proj[batch]             # (N, F)

        # ---- interaction blocks: 2-body then 3-body ----
        for conv2, conv3 in zip(self.two_body_convs,
                                self.three_body_convs):
            atom_fea = conv2(atom_fea, edge_fea, edge_index, cutoff_w)
            atom_fea = conv3(
                atom_fea, edge_fea, r_ij, dist,
                edge_index, triplet_idx, cutoff_w,
            )

        # ---- per-atom energy → total energy ----
        atom_energy = self.energy_head(atom_fea).squeeze(-1)     # (N,)
        energy = torch.zeros(batch_size, device=device, dtype=pos.dtype)
        energy.index_add_(0, batch, atom_energy)                 # (B,)

        results = {"energy": energy}

        # ---- forces ----
        if compute_forces:
            grad_pos = torch.autograd.grad(
                energy.sum(), pos,
                create_graph=self.training,
                retain_graph=True,
            )[0]
            results["forces"] = -grad_pos                        # (N, 3)

        # ---- stress via strain derivative ----
        if compute_stress:
            grad_strain = torch.autograd.grad(
                energy.sum(), strain,
                create_graph=self.training,
                retain_graph=True,
            )[0]                                                 # (B, 3, 3)
            # σ = (1/V) dE/dε   [eV / Å³]
            stress = grad_strain / volume.view(-1, 1, 1)
            # symmetrise (should already be ~symmetric, but numerics)
            stress = 0.5 * (stress + stress.transpose(1, 2))
            results["stress"] = stress                           # (B, 3, 3)

        return results
