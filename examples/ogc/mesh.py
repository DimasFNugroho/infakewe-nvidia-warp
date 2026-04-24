"""mesh.py — Triangulated cylinder + topology / feature normals for OGC.

Builds a Z-aligned capped cylinder as a triangle mesh, then extracts:

  - unique edges and which triangles are adjacent to each edge
  - per-feature outward normals used by the simplified feasibility gate:
        face_normals[t]     outward normal of triangle t
        edge_normals[e]     averaged outward normal at edge e
        vert_normals[v]     averaged outward normal at vertex v

For a convex obstacle (like a cylinder) these three arrays are sufficient to
implement the Gauss-map feasibility check from Sec. 4.1/4.2 of the paper in a
numerically stable way:  a candidate direction d is *feasible* against a
feature f when  dot(d, normal[f]) > 0, i.e. it points outward from the feature.
"""

from __future__ import annotations

import numpy as np


# ── Data container ────────────────────────────────────────────────────────────

class OGCMesh:
    """Triangle mesh + feature adjacency and outward normals."""

    def __init__(
        self,
        V: np.ndarray,          # (Nv, 3)  float32
        T: np.ndarray,          # (Nt, 3)  int32  (vertex indices per triangle)
        E: np.ndarray,          # (Ne, 2)  int32  (v0 < v1)
        tri_edges: np.ndarray,  # (Nt, 3)  int32  (edge index per tri side: e01, e12, e20)
        face_normals: np.ndarray,   # (Nt, 3) float32
        edge_normals: np.ndarray,   # (Ne, 3) float32
        vert_normals: np.ndarray,   # (Nv, 3) float32
    ):
        self.V = V.astype(np.float32, copy=False)
        self.T = T.astype(np.int32,   copy=False)
        self.E = E.astype(np.int32,   copy=False)
        self.tri_edges    = tri_edges.astype(np.int32, copy=False)
        self.face_normals = face_normals.astype(np.float32, copy=False)
        self.edge_normals = edge_normals.astype(np.float32, copy=False)
        self.vert_normals = vert_normals.astype(np.float32, copy=False)

    @property
    def num_vertices(self) -> int: return self.V.shape[0]

    @property
    def num_triangles(self) -> int: return self.T.shape[0]

    @property
    def num_edges(self) -> int: return self.E.shape[0]


# ── Cylinder builder ──────────────────────────────────────────────────────────

def build_cylinder(
    cx: float, cy: float, cz: float,
    radius: float,
    half_h: float,
    n_segs: int = 48,
) -> OGCMesh:
    """Build a Z-aligned capped cylinder as an OGCMesh.

    Vertex layout
    -------------
        [0]                       bottom-cap centre
        [1        .. n_segs]      bottom ring
        [n_segs+1 .. 2*n_segs]    top ring
        [2*n_segs+1]              top-cap centre
    """
    angles = np.linspace(0.0, 2.0 * np.pi, n_segs, endpoint=False, dtype=np.float32)
    ca, sa = np.cos(angles), np.sin(angles)

    bot_ring = np.column_stack([cx + radius * ca, cy + radius * sa,
                                np.full(n_segs, cz - half_h, dtype=np.float32)])
    top_ring = np.column_stack([cx + radius * ca, cy + radius * sa,
                                np.full(n_segs, cz + half_h, dtype=np.float32)])
    bot_ctr  = np.array([[cx, cy, cz - half_h]], dtype=np.float32)
    top_ctr  = np.array([[cx, cy, cz + half_h]], dtype=np.float32)

    V = np.vstack([bot_ctr, bot_ring, top_ring, top_ctr]).astype(np.float32)

    faces = []
    # Bottom cap — outward normal -Z
    for j in range(n_segs):
        faces.append([0, 1 + (j + 1) % n_segs, 1 + j])
    # Side quads — outward normal radial
    for j in range(n_segs):
        b0 = 1 + j;           b1 = 1 + (j + 1) % n_segs
        t0 = n_segs + 1 + j;  t1 = n_segs + 1 + (j + 1) % n_segs
        faces.append([b0, b1, t0])
        faces.append([b1, t1, t0])
    # Top cap — outward normal +Z
    tc = 2 * n_segs + 1
    for j in range(n_segs):
        faces.append([tc, n_segs + 1 + j, n_segs + 1 + (j + 1) % n_segs])
    T = np.array(faces, dtype=np.int32)

    E, tri_edges                       = _extract_edges(T)
    face_normals                       = _face_normals(V, T)
    edge_normals, vert_normals         = _feature_normals(V, T, E, face_normals)

    return OGCMesh(V, T, E, tri_edges, face_normals, edge_normals, vert_normals)


# ── Topology helpers ──────────────────────────────────────────────────────────

def _extract_edges(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return unique edge list E (Ne,2) and per-triangle edge indices (Nt,3).

    Per-triangle ordering: edge 0 = (v0,v1), edge 1 = (v1,v2), edge 2 = (v2,v0).
    """
    Nt = T.shape[0]
    tri_sides = np.stack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]], axis=1)  # (Nt,3,2)

    keys = np.sort(tri_sides.reshape(-1, 2), axis=1)                           # canonical (v0<v1)
    # Deduplicate by packing (v0, v1) into a single int64 key
    packed = keys[:, 0].astype(np.int64) * (T.max() + 1) + keys[:, 1].astype(np.int64)
    uniq, inverse = np.unique(packed, return_inverse=True)

    E = np.empty((len(uniq), 2), dtype=np.int32)
    # Pick first occurrence of each unique edge
    first_idx = np.empty(len(uniq), dtype=np.int64)
    seen      = np.full(len(uniq), -1, dtype=np.int64)
    for flat_i, u in enumerate(inverse):
        if seen[u] < 0:
            seen[u] = flat_i
    first_idx = seen
    E = keys[first_idx]

    tri_edges = inverse.reshape(Nt, 3).astype(np.int32)
    return E, tri_edges


def _face_normals(V: np.ndarray, T: np.ndarray) -> np.ndarray:
    a = V[T[:, 0]]; b = V[T[:, 1]]; c = V[T[:, 2]]
    n = np.cross(b - a, c - a)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    return (n / norm).astype(np.float32)


def _feature_normals(
    V: np.ndarray, T: np.ndarray, E: np.ndarray, face_normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-edge and per-vertex averaged outward normals."""
    Nv = V.shape[0]; Ne = E.shape[0]

    edge_acc  = np.zeros((Ne, 3), dtype=np.float32)
    vert_acc  = np.zeros((Nv, 3), dtype=np.float32)

    # Packed key lookup from canonical (v0<v1) → edge index
    max_v = int(T.max()) + 1
    edge_key_to_idx: dict[int, int] = {
        int(e[0]) * max_v + int(e[1]): i for i, e in enumerate(E)
    }

    for t_idx, tri in enumerate(T):
        nrm = face_normals[t_idx]
        for side in ((0, 1), (1, 2), (2, 0)):
            a = int(tri[side[0]]); b = int(tri[side[1]])
            if a > b: a, b = b, a
            edge_acc[edge_key_to_idx[a * max_v + b]] += nrm
        for vi in tri:
            vert_acc[int(vi)] += nrm

    edge_normals = _normalize_rows(edge_acc)
    vert_normals = _normalize_rows(vert_acc)
    return edge_normals, vert_normals


def _normalize_rows(X: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm[nrm < 1e-12] = 1.0
    return (X / nrm).astype(np.float32)


# ── Mesh rendering helpers ────────────────────────────────────────────────────

def mesh_for_render(m: OGCMesh) -> tuple[np.ndarray, np.ndarray]:
    """Return (vertices, faces) arrays in the shape vispy.visuals.Mesh expects."""
    return m.V, m.T.astype(np.uint32)
