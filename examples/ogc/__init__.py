"""Offset Geometric Contact (OGC) modules for the NVIDIA Warp yarn simulator.

Paper reference
---------------
    Anka He Chen et al., "Offset Geometric Contact", SIGGRAPH 2025.

This package provides Warp-backed implementations of the paper's four
algorithms, specialised to a yarn (polyline) colliding with a static
triangulated obstacle (e.g. a cylinder).

    algorithm1.py   Vertex-Facet contact detection (yarn vertices vs obstacle triangles)
    algorithm2.py   Edge-Edge   contact detection (yarn edges    vs obstacle edges)
    algorithm3.py   Simulation step orchestrator
    algorithm4.py   Inner solver iteration (PBD stretch + bend + OGC contact projection)

Supporting modules
------------------
    mesh.py         Cylinder mesh builder + topology (edges, adjacency, feature normals)

Simplifications relative to the paper (documented per-module)
-------------------------------------------------------------
  - Brute-force O(Nv * Nt) contact detection — no BVH.  Fine for a ~150-tri cylinder.
  - Simplified feasibility gate suitable for convex obstacles (cylinder): direction
    must point along a precomputed per-feature outward normal.  See mesh.py.
  - The inner solver is PBD (matching the parent project) rather than VBD.  The
    algorithmic *structure* of Algorithm 4 is preserved: one iteration = solve all
    constraints (stretch + bend + contact) in turn, color-free because yarn
    constraints already use even/odd splitting.
  - Conservative bounds + re-detection trigger from Algorithm 3 are dropped; contact
    detection runs every substep.  This is more conservative (slower) but simpler.
"""
