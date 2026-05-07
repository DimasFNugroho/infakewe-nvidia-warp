"""algorithm5.py — Yarn self-collision edge-edge detection.

For every yarn edge i, find the closest other yarn edge j (|i − j| > skip_adj)
within contact radius r.  Stores at most one contact per yarn edge (the closest
one) in SelfEEContacts.

Wound-section skip: pairs where both i and j are fully within the wound region
(both < n_wound - 1) are skipped.  Adjacent wound wraps are always at contact
distance by design and would create persistent projection forces that corrupt
the strain field.  Free-span vs. wound contacts are still detected.

Output arrays indexed by yarn edge i (0 .. N-2):
    active[i]    0/1
    cp_self[i]   closest point on edge i
    cp_other[i]  closest point on closest edge j
    dist[i]      distance between them
    normal[i]    (cp_self − cp_other).normalized()  — points from j toward i
    s[i]         parametric position along edge i (for barycentric correction)
"""

from __future__ import annotations

import warp as wp

from .algorithm2 import _edge_edge_closest   # reuse the @wp.func segment helper


@wp.kernel
def kernel_self_ee_detect(
    pos:        wp.array(dtype=wp.vec3),
    yarn_edges: wp.array(dtype=wp.vec2i),
    r:          float,
    skip_adj:   int,
    n_wound:    int,
    # outputs:
    active:     wp.array(dtype=int),
    cp_self:    wp.array(dtype=wp.vec3),
    cp_other:   wp.array(dtype=wp.vec3),
    dist_out:   wp.array(dtype=float),
    normal_out: wp.array(dtype=wp.vec3),
    s_out:      wp.array(dtype=float),
):
    i = wp.tid()
    ev_i = yarn_edges[i]
    p1   = pos[ev_i[0]]
    q1   = pos[ev_i[1]]

    best_dist   = float(r)
    best_cp1    = wp.vec3(float(0.0), float(0.0), float(0.0))
    best_cp2    = wp.vec3(float(0.0), float(0.0), float(0.0))
    best_normal = wp.vec3(float(0.0), float(1.0), float(0.0))
    best_s      = float(0.5)
    found       = int(0)

    n_wound_edge = n_wound - 1   # last fully-wound edge index (exclusive)

    n_edges = yarn_edges.shape[0]
    for j in range(n_edges):
        # Skip pairs where both edges are entirely within the wound section.
        if i < n_wound_edge and j < n_wound_edge:
            continue

        diff = i - j
        if diff < 0:
            diff = -diff
        if diff <= skip_adj:
            continue

        ev_j = yarn_edges[j]
        p2   = pos[ev_j[0]]
        q2   = pos[ev_j[1]]

        cp1, cp2, s, t = _edge_edge_closest(p1, q1, p2, q2)
        d = wp.length(cp1 - cp2)
        if d >= best_dist:
            continue

        direction = cp1 - cp2
        if wp.length(direction) < float(1.0e-12):
            continue
        direction = wp.normalize(direction)

        best_dist   = d
        best_cp1    = cp1
        best_cp2    = cp2
        best_normal = direction
        best_s      = s
        found       = int(1)

    active[i]     = found
    cp_self[i]    = best_cp1
    cp_other[i]   = best_cp2
    dist_out[i]   = best_dist
    normal_out[i] = best_normal
    s_out[i]      = best_s


class SelfEEContacts:
    """GPU-resident yarn-self-collision edge-edge contact arrays."""

    def __init__(self, n_yarn_edges: int, device: str):
        self.active   = wp.zeros(n_yarn_edges, dtype=int,     device=device)
        self.cp_self  = wp.zeros(n_yarn_edges, dtype=wp.vec3, device=device)
        self.cp_other = wp.zeros(n_yarn_edges, dtype=wp.vec3, device=device)
        self.dist     = wp.zeros(n_yarn_edges, dtype=float,   device=device)
        self.normal   = wp.zeros(n_yarn_edges, dtype=wp.vec3, device=device)
        self.s        = wp.zeros(n_yarn_edges, dtype=float,   device=device)


def detect_self_ee(
    pos:        wp.array,
    yarn_edges: wp.array,
    contacts:   SelfEEContacts,
    r:          float,
    device:     str,
    n_wound:    int = 0,
    skip_adj:   int = 2,
):
    """Run self-EE detection: fill `contacts` with the closest self-contact per yarn edge.

    n_wound: number of wound particles — wound-vs-wound edge pairs are skipped
             to avoid persistent contacts in the helically packed wound section.
    """
    wp.launch(
        kernel_self_ee_detect,
        dim=yarn_edges.shape[0],
        device=device,
        inputs=[
            pos, yarn_edges, float(r), int(skip_adj), int(n_wound),
            contacts.active, contacts.cp_self, contacts.cp_other,
            contacts.dist, contacts.normal, contacts.s,
        ],
    )
