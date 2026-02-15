"""
graph_utils.py
Crystal graph construction from pymatgen Structure objects.

Atoms are nodes, bonds (within a distance cutoff) are directed edges.
Periodic boundary conditions are handled by storing Cartesian offset
vectors for each edge so that distances remain differentiable through
atomic positions (required for autograd forces).

Edge convention  (2-body)
-------------------------
  src  = neighbour  (message sender)
  dst  = centre     (message receiver / aggregation target)
  offset = Cartesian translation that maps the unit-cell copy of the
           neighbour atom to its periodic image used for this edge.

  displacement:  r_ij = pos[src] + offset - pos[dst]

Triplet convention  (3-body)
----------------------------
  For every centre atom i with incoming edges  e1 = (j → i)  and
  e2 = (k → i),  we store an ordered pair  (e1, e2).
  The angle  θ_{j,i,k}  is the angle at centre i between
  displacement vectors r_ij and r_ik.

  triplet_idx[0]  =  "primary"   edge index  (e1)
  triplet_idx[1]  =  "secondary" edge index  (e2)
  Both edges share the same destination (centre) atom.
"""

import numpy as np
from pymatgen.core import Structure


# ---------------------------------------------------------------------------
#  2-body graph
# ---------------------------------------------------------------------------

def build_crystal_graph(
    structure: Structure,
    cutoff: float = 5.0,
    max_neighbors: int = 12,
):
    """
    Build a crystal graph from a pymatgen Structure.

    Parameters
    ----------
    structure : pymatgen.core.Structure
    cutoff : float
        Neighbour search radius in Angstroms.
    max_neighbors : int
        Maximum neighbours retained per centre atom (closest first).

    Returns
    -------
    dict with keys
        atom_types   : (N,)    int64   – atomic numbers
        positions    : (N, 3)  float64 – Cartesian coords  [Angstrom]
        lattice      : (3, 3)  float64 – lattice matrix     [Angstrom]
        edge_index   : (2, E)  int64   – [src, dst]
        edge_offset  : (E, 3)  float64 – Cartesian offsets   [Angstrom]
        triplet_idx  : (2, T)  int64   – [primary_edge, secondary_edge]
        num_atoms    : int
        num_edges    : int
        volume       : float                                 [Angstrom^3]
    """
    num_atoms = len(structure)
    all_nbrs = structure.get_all_neighbors(cutoff)

    src_list, dst_list, offset_list = [], [], []

    for centre_idx, nbrs in enumerate(all_nbrs):
        # sort by distance, keep closest max_neighbors
        nbrs_sorted = sorted(nbrs, key=lambda x: x.nn_distance)[:max_neighbors]

        for nbr in nbrs_sorted:
            neighbour_idx = nbr.index
            image = np.array(nbr.image, dtype=np.float64)  # fractional integers
            offset_cart = image @ structure.lattice.matrix   # Cartesian offset

            src_list.append(neighbour_idx)   # sender   = neighbour
            dst_list.append(centre_idx)      # receiver = centre
            offset_list.append(offset_cart)

    if len(src_list) == 0:
        raise ValueError(
            f"No edges found ({num_atoms} atoms, cutoff={cutoff} A). "
            "Increase cutoff or check the structure."
        )

    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    num_edges = edge_index.shape[1]

    # --- 3-body triplet indices ---
    triplet_idx = _compute_triplets(edge_index, num_atoms)

    return {
        "atom_types":   np.array([s.specie.Z for s in structure], dtype=np.int64),
        "positions":    np.array(structure.cart_coords, dtype=np.float64),
        "lattice":      np.array(structure.lattice.matrix, dtype=np.float64),
        "edge_index":   edge_index,
        "edge_offset":  np.array(offset_list, dtype=np.float64),
        "triplet_idx":  triplet_idx,
        "num_atoms":    num_atoms,
        "num_edges":    num_edges,
        "volume":       float(structure.volume),
    }


# ---------------------------------------------------------------------------
#  3-body triplet construction
# ---------------------------------------------------------------------------

def _compute_triplets(edge_index, num_atoms):
    """
    For every centre atom *i* that has ≥ 2 incoming edges, enumerate all
    ordered pairs  (e1, e2)  where  dst[e1] == dst[e2] == i  and e1 ≠ e2.

    Parameters
    ----------
    edge_index : (2, E)  int64 – [src, dst]
    num_atoms  : int

    Returns
    -------
    triplet_idx : (2, T) int64 – [primary_edge, secondary_edge]
    """
    dst = edge_index[1]

    # group edge indices by their destination (centre) atom
    edges_per_centre = [[] for _ in range(num_atoms)]
    for e_idx in range(edge_index.shape[1]):
        edges_per_centre[dst[e_idx]].append(e_idx)

    primary, secondary = [], []
    for centre_edges in edges_per_centre:
        n = len(centre_edges)
        if n < 2:
            continue
        for a in range(n):
            for b in range(n):
                if a != b:
                    primary.append(centre_edges[a])
                    secondary.append(centre_edges[b])

    if len(primary) == 0:
        # degenerate case – every atom has at most 1 neighbour
        return np.zeros((2, 0), dtype=np.int64)

    return np.array([primary, secondary], dtype=np.int64)
