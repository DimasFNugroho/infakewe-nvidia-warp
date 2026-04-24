"""algorithm1.py — Vertex-Facet contact detection (OGC Algorithm 1).

For every yarn vertex v, find the closest feature (face / edge / vertex) on
the obstacle mesh within contact radius r, and record the contact if it
passes the feasibility gate (Eq. 8, 9 of the paper — simplified here to a
per-feature outward-normal test, see mesh.py).

Output arrays (one slot per yarn vertex; inactive if vf_active[v] == 0):

    vf_active[v]     0 / 1            is a feasible contact recorded?
    vf_cp[v]         wp.vec3          closest point on obstacle
    vf_dist[v]       float            distance vertex → closest point
    vf_normal[v]     wp.vec3          outward unit normal at the closest feature
                                      (used by Algorithm 4 for projection)

We store at most one contact per yarn vertex (the closest feasible one);
for the simple yarn-vs-cylinder scene this is sufficient.  The paper's
FOGC(v) would be a set — extending this array to (N_yarn, max_contacts)
follows the same pattern.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from .mesh import OGCMesh


# Feature codes emitted by the kernel.
FEAT_FACE = wp.constant(0)
FEAT_EDGE = wp.constant(1)
FEAT_VERT = wp.constant(2)


# ── Point–triangle distance with closest feature classification ───────────────

@wp.func
def _closest_pt_triangle(
    p: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3,
):
    """Ericson-style closest-point-on-triangle.

    Returns (cp, feature_code, feature_local_idx).
      feature_code      0=face, 1=edge, 2=vertex
      feature_local_idx face: 0
                        edge: 0 (ab), 1 (bc), 2 (ca)
                        vert: 0 (a),  1 (b),  2 (c)
    """
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = wp.dot(ab, ap); d2 = wp.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a, FEAT_VERT, 0

    bp = p - b
    d3 = wp.dot(ab, bp); d4 = wp.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b, FEAT_VERT, 1

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + ab * v, FEAT_EDGE, 0  # edge ab

    cp_ = p - c
    d5 = wp.dot(ab, cp_); d6 = wp.dot(ac, cp_)
    if d6 >= 0.0 and d5 <= d6:
        return c, FEAT_VERT, 2

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + ac * w, FEAT_EDGE, 2  # edge ca

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + (c - b) * w, FEAT_EDGE, 1  # edge bc

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return a + ab * v + ac * w, FEAT_FACE, 0


# ── Main VF detection kernel ──────────────────────────────────────────────────

@wp.kernel
def kernel_vf_detect(
    yarn_pos:     wp.array(dtype=wp.vec3),
    obs_V:        wp.array(dtype=wp.vec3),
    obs_T:        wp.array(dtype=wp.vec3i),      # per-tri vertex indices
    obs_tri_edges: wp.array(dtype=wp.vec3i),     # per-tri global edge indices
    face_normals: wp.array(dtype=wp.vec3),
    edge_normals: wp.array(dtype=wp.vec3),
    vert_normals: wp.array(dtype=wp.vec3),
    r:            float,                         # contact radius
    # Outputs (one slot per yarn vertex):
    vf_active:    wp.array(dtype=int),
    vf_cp:        wp.array(dtype=wp.vec3),
    vf_dist:      wp.array(dtype=float),
    vf_normal:    wp.array(dtype=wp.vec3),
):
    v_idx = wp.tid()
    p     = yarn_pos[v_idx]

    best_dist   = float(r)
    best_cp     = wp.vec3(0.0, 0.0, 0.0)
    best_normal = wp.vec3(0.0, 1.0, 0.0)
    found       = int(0)

    n_tris = obs_T.shape[0]
    for t in range(n_tris):
        tri = obs_T[t]
        a   = obs_V[tri[0]]
        b   = obs_V[tri[1]]
        c   = obs_V[tri[2]]

        cp, feat, local = _closest_pt_triangle(p, a, b, c)
        d = wp.length(p - cp)
        if d >= best_dist:
            continue

        # Resolve the feature's outward normal
        if feat == FEAT_FACE:
            n_feat = face_normals[t]
        elif feat == FEAT_EDGE:
            n_feat = edge_normals[obs_tri_edges[t][local]]
        else:
            n_feat = vert_normals[tri[local]]

        # Feasibility: direction from obstacle feature → yarn vertex must point outward.
        direction = p - cp
        if wp.dot(direction, n_feat) <= 0.0:
            continue

        best_dist   = d
        best_cp     = cp
        best_normal = n_feat
        found       = int(1)

    vf_active[v_idx] = found
    vf_cp[v_idx]     = best_cp
    vf_dist[v_idx]   = best_dist
    vf_normal[v_idx] = best_normal


# ── Python entry point ────────────────────────────────────────────────────────

class VFContacts:
    """GPU-resident vertex-facet contact arrays."""

    def __init__(self, n_yarn: int, device: str):
        self.active = wp.zeros(n_yarn, dtype=int,     device=device)
        self.cp     = wp.zeros(n_yarn, dtype=wp.vec3, device=device)
        self.dist   = wp.zeros(n_yarn, dtype=float,   device=device)
        self.normal = wp.zeros(n_yarn, dtype=wp.vec3, device=device)


class ObstacleGPU:
    """Immutable obstacle arrays uploaded to the Warp device once."""

    def __init__(self, mesh: OGCMesh, device: str):
        self.V            = wp.array(mesh.V,            dtype=wp.vec3,  device=device)
        self.T            = wp.array(mesh.T,            dtype=wp.vec3i, device=device)
        self.tri_edges    = wp.array(mesh.tri_edges,    dtype=wp.vec3i, device=device)
        self.E            = wp.array(mesh.E,            dtype=wp.vec2i, device=device)
        self.face_normals = wp.array(mesh.face_normals, dtype=wp.vec3,  device=device)
        self.edge_normals = wp.array(mesh.edge_normals, dtype=wp.vec3,  device=device)
        self.vert_normals = wp.array(mesh.vert_normals, dtype=wp.vec3,  device=device)


def detect_vertex_facet(
    yarn_pos: wp.array,
    obstacle: ObstacleGPU,
    contacts: VFContacts,
    r: float,
    device: str,
):
    """Run Algorithm 1: fill `contacts` with at most one closest feasible
    obstacle feature per yarn vertex.
    """
    wp.launch(
        kernel_vf_detect,
        dim=yarn_pos.shape[0],
        device=device,
        inputs=[
            yarn_pos,
            obstacle.V, obstacle.T, obstacle.tri_edges,
            obstacle.face_normals, obstacle.edge_normals, obstacle.vert_normals,
            r,
            contacts.active, contacts.cp, contacts.dist, contacts.normal,
        ],
    )
