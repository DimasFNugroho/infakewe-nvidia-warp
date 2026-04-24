"""algorithm2.py — Edge-Edge contact detection (OGC Algorithm 2).

For every yarn edge e = (v_a, v_b), find the closest obstacle edge within
contact radius r whose closest-point pair lies strictly in the interior of
both edges (s, t ∈ (0, 1)).  Contacts at edge endpoints are already handled
by Algorithm 1 as vertex–edge contacts.

Output arrays (one slot per yarn edge; inactive if ee_active[e] == 0):

    ee_active[e]     0 / 1
    ee_cp_yarn[e]    wp.vec3   closest point on the yarn edge
    ee_cp_obs[e]     wp.vec3   closest point on the obstacle edge
    ee_dist[e]       float     distance between them
    ee_normal[e]     wp.vec3   mutual normal ≈ (cp_yarn - cp_obs).normalized()
    ee_s[e]          float     parametric position along the yarn edge (0..1)
"""

from __future__ import annotations

import numpy as np
import warp as wp

from .algorithm1 import ObstacleGPU


# ── Edge–edge closest points ──────────────────────────────────────────────────

@wp.func
def _edge_edge_closest(
    p1: wp.vec3, q1: wp.vec3,
    p2: wp.vec3, q2: wp.vec3,
):
    """Shortest segment between two line segments (p1,q1) and (p2,q2).
    Returns (cp1, cp2, s, t).  Interior contact = 0 < s,t < 1.
    """
    d1 = q1 - p1
    d2 = q2 - p2
    r  = p1 - p2
    a  = wp.dot(d1, d1)
    e  = wp.dot(d2, d2)
    f  = wp.dot(d2, r)

    s = float(0.0); t = float(0.0)
    eps = 1.0e-12

    if a <= eps and e <= eps:
        return p1, p2, 0.0, 0.0

    if a <= eps:
        s = 0.0
        t = wp.clamp(f / e, 0.0, 1.0)
    else:
        c = wp.dot(d1, r)
        if e <= eps:
            t = 0.0
            s = wp.clamp(-c / a, 0.0, 1.0)
        else:
            b     = wp.dot(d1, d2)
            denom = a * e - b * b
            if denom != 0.0:
                s = wp.clamp((b * f - c * e) / denom, 0.0, 1.0)
            else:
                s = 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = wp.clamp(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = wp.clamp((b - c) / a, 0.0, 1.0)

    cp1 = p1 + d1 * s
    cp2 = p2 + d2 * t
    return cp1, cp2, s, t


# ── Main EE detection kernel ──────────────────────────────────────────────────

@wp.kernel
def kernel_ee_detect(
    yarn_pos:      wp.array(dtype=wp.vec3),
    yarn_edges:    wp.array(dtype=wp.vec2i),   # (Ne_yarn, 2) global yarn vertex indices
    obs_V:         wp.array(dtype=wp.vec3),
    obs_E:         wp.array(dtype=wp.vec2i),
    edge_normals:  wp.array(dtype=wp.vec3),
    r:             float,
    interior_eps:  float,                      # margin so we strictly exclude endpoints
    # Outputs per yarn edge:
    ee_active:     wp.array(dtype=int),
    ee_cp_yarn:    wp.array(dtype=wp.vec3),
    ee_cp_obs:     wp.array(dtype=wp.vec3),
    ee_dist:       wp.array(dtype=float),
    ee_normal:     wp.array(dtype=wp.vec3),
    ee_s:          wp.array(dtype=float),
):
    i   = wp.tid()
    ev  = yarn_edges[i]
    p1  = yarn_pos[ev[0]]
    q1  = yarn_pos[ev[1]]

    best_dist   = float(r)
    best_cp1    = wp.vec3(0.0, 0.0, 0.0)
    best_cp2    = wp.vec3(0.0, 0.0, 0.0)
    best_normal = wp.vec3(0.0, 1.0, 0.0)
    best_s      = float(0.5)
    found       = int(0)

    n_obs_e = obs_E.shape[0]
    for j in range(n_obs_e):
        oe = obs_E[j]
        p2 = obs_V[oe[0]]
        q2 = obs_V[oe[1]]

        cp1, cp2, s, t = _edge_edge_closest(p1, q1, p2, q2)

        # Require interior contact on both segments (endpoints handled by Alg 1).
        if s <= interior_eps or s >= 1.0 - interior_eps:
            continue
        if t <= interior_eps or t >= 1.0 - interior_eps:
            continue

        d = wp.length(cp1 - cp2)
        if d >= best_dist:
            continue

        # Feasibility: mutual normal must point outward from the obstacle edge.
        direction = cp1 - cp2
        if wp.length(direction) < 1.0e-12:
            continue
        direction = wp.normalize(direction)

        if wp.dot(direction, edge_normals[j]) <= 0.0:
            continue

        best_dist   = d
        best_cp1    = cp1
        best_cp2    = cp2
        best_normal = direction
        best_s      = s
        found       = int(1)

    ee_active[i]  = found
    ee_cp_yarn[i] = best_cp1
    ee_cp_obs[i]  = best_cp2
    ee_dist[i]    = best_dist
    ee_normal[i]  = best_normal
    ee_s[i]       = best_s


# ── Python entry point ────────────────────────────────────────────────────────

class EEContacts:
    """GPU-resident edge-edge contact arrays."""

    def __init__(self, n_yarn_edges: int, device: str):
        self.active  = wp.zeros(n_yarn_edges, dtype=int,     device=device)
        self.cp_yarn = wp.zeros(n_yarn_edges, dtype=wp.vec3, device=device)
        self.cp_obs  = wp.zeros(n_yarn_edges, dtype=wp.vec3, device=device)
        self.dist    = wp.zeros(n_yarn_edges, dtype=float,   device=device)
        self.normal  = wp.zeros(n_yarn_edges, dtype=wp.vec3, device=device)
        self.s       = wp.zeros(n_yarn_edges, dtype=float,   device=device)


def detect_edge_edge(
    yarn_pos:   wp.array,
    yarn_edges: wp.array,
    obstacle:   ObstacleGPU,
    contacts:   EEContacts,
    r:          float,
    device:     str,
    interior_eps: float = 1.0e-3,
):
    wp.launch(
        kernel_ee_detect,
        dim=yarn_edges.shape[0],
        device=device,
        inputs=[
            yarn_pos, yarn_edges,
            obstacle.V, obstacle.E, obstacle.edge_normals,
            r, interior_eps,
            contacts.active, contacts.cp_yarn, contacts.cp_obs,
            contacts.dist,  contacts.normal,   contacts.s,
        ],
    )
