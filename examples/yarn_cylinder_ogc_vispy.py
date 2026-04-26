"""examples/yarn_cylinder_ogc_vispy.py — Yarn-vs-cylinder using OGC on Warp.

Same scene as `yarn_cylinder_vispy.py` (free-fall yarn, pinned at one end,
colliding with a Z-aligned cylinder), but the collision is resolved by the
Offset Geometric Contact (OGC) pipeline instead of the analytic
cylinder-projection kernel.

Pipeline (all four algorithms from OGC / SIGGRAPH 2025):

    examples.ogc.algorithm1   Vertex-Facet detection (Warp kernel)
    examples.ogc.algorithm2   Edge-Edge   detection (Warp kernel)
    examples.ogc.algorithm3   Simulation step orchestrator
    examples.ogc.algorithm4   Inner iteration — stretch + bend + OGC projection

Because OGC treats the cylinder as a triangulated mesh *plus* an offset
radius r, the yarn collides with a smooth "inflated" version of the mesh.
Small r recovers the original cylinder; larger r widens the contact band
and makes seam effects visible.

Controls
--------
  Space       — pause / resume
  R           — reset simulation
  Left-drag   — orbit camera
  Scroll      — zoom
  Middle-drag — pan

Run
---
    python examples/yarn_cylinder_ogc_vispy.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import warp as wp

from vispy import app, scene
from vispy.scene import visuals

from ogc.mesh       import build_cylinder, mesh_for_render
from ogc.algorithm3 import OGCSimulation


# ── Cylinder parameters (match yarn_cylinder_vispy.py) ────────────────────────
CYL_CENTER_X   =  1.0
CYL_CENTER_Y   = -0.55
CYL_CENTER_Z   =  0.0
CYL_RADIUS     =  0.22
CYL_HALF_H     =  1.5
CYL_N_SEGS     =  48

# ── OGC parameters ────────────────────────────────────────────────────────────
#   r = contact radius of the offset geometry around the cylinder.
#   Small values make the yarn hug the triangulated surface; larger values
#   behave like a thicker smoothed cylinder.
OGC_CONTACT_RADIUS    = 0.02
OGC_CONTACT_STIFFNESS = 1.0

# ── Colours ───────────────────────────────────────────────────────────────────
BG_COL     = (0.07, 0.07, 0.13, 1.0)
ANCHOR_COL = (0.0,  0.83, 1.0)
FREE_COL   = (1.0,  0.65, 0.0)
CYL_COL    = (0.25, 0.70, 0.45, 0.75)

# Repeating stripe palette — each band is STRIPE_SIZE consecutive particles.
# Fixed to particle index so you can track individual segments visually.
STRIPE_SIZE    = 5
STRIPE_PALETTE = np.array([
    [0.95, 0.25, 0.35, 1.0],   # red
    [0.98, 0.80, 0.12, 1.0],   # gold
    [0.18, 0.82, 0.95, 1.0],   # cyan
    [0.65, 0.40, 0.95, 1.0],   # violet
], dtype=np.float32)


def make_yarn_colors(n: int) -> np.ndarray:
    """Return (n, 4) RGBA per-vertex colour array with repeating stripes."""
    palette_n = len(STRIPE_PALETTE)
    idx = (np.arange(n) // STRIPE_SIZE) % palette_n
    return STRIPE_PALETTE[idx]


# ── Vispy visualizer ──────────────────────────────────────────────────────────

class CylinderOGCVispy:
    def __init__(self, sim: OGCSimulation, cyl_mesh, fps: int = 60):
        self.sim      = sim
        self.cyl_mesh = cyl_mesh
        self.paused   = False
        self.frame    = 0
        self._build_scene()
        self._timer = app.Timer(interval=1.0 / fps, connect=self._on_timer, start=True)

    def _build_scene(self):
        self.canvas = scene.SceneCanvas(
            title="Yarn + Cylinder — OGC (Offset Geometric Contact) on Warp",
            size=(960, 720),
            bgcolor=BG_COL,
            keys="interactive",
            show=True,
        )
        self.canvas.events.key_press.connect(self._on_key)

        view = self.canvas.central_widget.add_view()
        view.camera = scene.cameras.TurntableCamera(
            elevation=20, azimuth=-60, distance=5.5, up="y", fov=45,
        )
        visuals.XYZAxis(parent=view.scene)

        # ── Cylinder rendered from the OGC mesh itself ────────────────────────
        verts, faces = mesh_for_render(self.cyl_mesh)
        visuals.Mesh(
            vertices=verts, faces=faces, color=CYL_COL, shading="smooth",
            parent=view.scene,
        )

        # ── Yarn ──────────────────────────────────────────────────────────────
        p = self.sim.positions()
        self._yarn_colors = make_yarn_colors(p.shape[0])
        self._yarn = visuals.Line(
            pos=p, color=self._yarn_colors, width=3,
            connect="strip", parent=view.scene,
        )
        self._anchor   = visuals.Markers(parent=view.scene)
        self._free_end = visuals.Markers(parent=view.scene)
        self._anchor.set_data(p[:1],   face_color=ANCHOR_COL, size=14, edge_width=0)
        self._free_end.set_data(p[-1:], face_color=FREE_COL,   size=14, edge_width=0)

        # ── HUD ───────────────────────────────────────────────────────────────
        self._hud = visuals.Text(
            text=self._hud_text(),
            color="white", font_size=10,
            anchor_x="left", anchor_y="top",
            parent=self.canvas.scene, pos=(10, 20),
        )

    def _on_timer(self, _event):
        if not self.paused:
            self.sim.step()
            self.frame += 1
        p = self.sim.positions()
        self._yarn.set_data(pos=p)
        self._anchor.set_data(p[:1],   face_color=ANCHOR_COL, size=14, edge_width=0)
        self._free_end.set_data(p[-1:], face_color=FREE_COL,   size=14, edge_width=0)
        self._hud.text = self._hud_text()
        self.canvas.update()

    def _on_key(self, event):
        if event.key == "Space":
            self.paused = not self.paused
        elif event.key == "R":
            self.sim.reset()
            self.frame = 0

    def _hud_text(self) -> str:
        status = "PAUSED" if self.paused else f"t={self.sim.time:.2f}s"
        return (
            f"OGC yarn–cylinder  |  frame {self.frame:04d}  {status}  "
            f"[{self.sim.device}]\n"
            f"r={self.sim.r:g}  verts={self.cyl_mesh.num_vertices}  "
            f"tris={self.cyl_mesh.num_triangles}  "
            f"edges={self.cyl_mesh.num_edges}\n"
            "Space=pause  R=reset  drag=orbit  scroll=zoom"
        )

    def run(self):
        app.run()


# ── Main ──────────────────────────────────────────────────────────────────────

wp.init()

cyl = build_cylinder(
    CYL_CENTER_X, CYL_CENTER_Y, CYL_CENTER_Z,
    CYL_RADIUS, CYL_HALF_H, n_segs=CYL_N_SEGS,
)

sim = OGCSimulation(
    obstacle_mesh     = cyl,
    contact_radius    = OGC_CONTACT_RADIUS,
    contact_stiffness = OGC_CONTACT_STIFFNESS,
)

print(f"OGC yarn demo — {sim.device}, yarn={sim.pos.shape[0]} particles, "
      f"cylinder={cyl.num_triangles} tris / {cyl.num_edges} edges, "
      f"r={OGC_CONTACT_RADIUS}")

viz = CylinderOGCVispy(sim, cyl)
viz.run()
