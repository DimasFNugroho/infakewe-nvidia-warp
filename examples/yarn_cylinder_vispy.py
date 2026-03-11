"""examples/yarn_cylinder_vispy.py — Free-fall yarn colliding with a cylinder, via Vispy.

A horizontal yarn falls under gravity onto a static cylinder that is
oriented along the Z axis.  Collision is handled by a custom Warp kernel
that pushes particles outside the cylinder surface.

Controls
--------
  Space       — pause / resume
  R           — reset simulation
  Left-drag   — orbit camera
  Scroll      — zoom
  Middle-drag — pan

Run
---
    python examples/yarn_cylinder_vispy.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import warp as wp

from vispy import app, scene
from vispy.scene import visuals

import config
from simulation import Simulation


# ── Cylinder parameters ───────────────────────────────────────────────────────
CYL_CENTER_X   =  1.0    # midpoint of the yarn along X
CYL_CENTER_Y   = -0.55   # below the yarn's starting height
CYL_CENTER_Z   =  0.0
CYL_RADIUS     =  0.22
CYL_HALF_H     =  1.5    # half-height along Z (wider than the yarn in depth)

# ── Colours ───────────────────────────────────────────────────────────────────
BG_COL     = (0.07, 0.07, 0.13, 1.0)
YARN_COL   = (0.91, 0.27, 0.38)
ANCHOR_COL = (0.0,  0.83, 1.0)
FREE_COL   = (1.0,  0.65, 0.0)
CYL_COL    = (0.25, 0.70, 0.45, 0.75)   # semi-transparent green


# ── Collision kernel ──────────────────────────────────────────────────────────

@wp.kernel
def kernel_cylinder_collision(
    pos:        wp.array(dtype=wp.vec3),
    inv_mass:   wp.array(dtype=float),
    cyl_center: wp.vec3,
    radius:     float,
    half_h:     float,
):
    """Push any particle that penetrates the Z-aligned cylinder back to its surface."""
    i = wp.tid()
    if inv_mass[i] == 0.0:
        return

    p  = pos[i]
    dx = p[0] - cyl_center[0]
    dy = p[1] - cyl_center[1]
    dz = p[2] - cyl_center[2]

    # Only act within the finite length of the cylinder
    if wp.abs(dz) > half_h:
        return

    r = wp.sqrt(dx * dx + dy * dy)
    if r >= radius:
        return   # outside — no collision

    # Project radially onto the cylinder surface
    if r < 1.0e-8:
        # Degenerate: particle sits on the axis; push in +X
        pos[i] = wp.vec3(cyl_center[0] + radius, p[1], p[2])
    else:
        s      = radius / r
        pos[i] = wp.vec3(cyl_center[0] + dx * s,
                         cyl_center[1] + dy * s,
                         p[2])


# ── Contact damping kernel ────────────────────────────────────────────────────

@wp.kernel
def kernel_cylinder_contact_damping(
    pos:        wp.array(dtype=wp.vec3),
    vel:        wp.array(dtype=wp.vec3),
    inv_mass:   wp.array(dtype=float),
    cyl_center: wp.vec3,
    radius:     float,
    half_h:     float,
    skin:       float,    # thin shell outside the surface that counts as "contact"
):
    """Zero the inward radial velocity component for particles resting on the cylinder.

    Without this, the PBD position correction each substep creates an outward
    velocity via kernel_update_velocity, which prevents the yarn from settling.
    """
    i = wp.tid()
    if inv_mass[i] == 0.0:
        return

    p  = pos[i]
    dx = p[0] - cyl_center[0]
    dy = p[1] - cyl_center[1]
    dz = p[2] - cyl_center[2]

    if wp.abs(dz) > half_h:
        return

    r = wp.sqrt(dx * dx + dy * dy)
    # Act on particles within the contact skin band around the surface
    if r > radius + skin or r < 1.0e-8:
        return

    # Radial unit normal pointing outward from the cylinder axis
    nx = dx / r
    ny = dy / r

    # Remove any inward (negative radial) velocity component
    v      = vel[i]
    v_rad  = v[0] * nx + v[1] * ny   # radial velocity (positive = outward)
    if v_rad < 0.0:
        vel[i] = wp.vec3(v[0] - v_rad * nx,
                         v[1] - v_rad * ny,
                         v[2])


# ── Simulation with cylinder collision ───────────────────────────────────────

class YarnCylinderSim(Simulation):
    def _initial_arrays(self):
        n      = config.NUM_PARTICLES
        pos_np = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            pos_np[i] = [i * config.REST_LEN, 0.0, 0.0]
        return pos_np, np.zeros((n, 3), dtype=np.float32)

    def _wind(self) -> wp.vec3:
        return wp.vec3(0.0, 0.0, 0.0)

    def _solve_constraints(self):
        """Run standard PBD constraints then enforce cylinder non-penetration."""
        super()._solve_constraints()
        wp.launch(
            kernel_cylinder_collision,
            dim=config.NUM_PARTICLES,
            device=self.device,
            inputs=[
                self.pos, self.inv_mass,
                wp.vec3(CYL_CENTER_X, CYL_CENTER_Y, CYL_CENTER_Z),
                CYL_RADIUS, CYL_HALF_H,
            ],
        )

    def _correct_velocity(self, inv_sdt: float):
        """Derive velocity from position delta, then damp contact velocities."""
        super()._correct_velocity(inv_sdt)
        wp.launch(
            kernel_cylinder_contact_damping,
            dim=config.NUM_PARTICLES,
            device=self.device,
            inputs=[
                self.pos, self.vel, self.inv_mass,
                wp.vec3(CYL_CENTER_X, CYL_CENTER_Y, CYL_CENTER_Z),
                CYL_RADIUS, CYL_HALF_H,
                0.01,   # skin thickness (m)
            ],
        )


# ── Cylinder mesh geometry ────────────────────────────────────────────────────

def _make_cylinder_mesh(cx, cy, cz, radius, half_h, n_segs=48):
    """Return (vertices, faces) for a Z-aligned cylinder as a triangle mesh."""
    angles  = np.linspace(0.0, 2.0 * np.pi, n_segs, endpoint=False, dtype=np.float32)
    cos_a   = np.cos(angles)
    sin_a   = np.sin(angles)

    bot_ring = np.column_stack([cx + radius * cos_a,
                                cy + radius * sin_a,
                                np.full(n_segs, cz - half_h, dtype=np.float32)])
    top_ring = np.column_stack([cx + radius * cos_a,
                                cy + radius * sin_a,
                                np.full(n_segs, cz + half_h, dtype=np.float32)])
    bot_ctr  = np.array([[cx, cy, cz - half_h]], dtype=np.float32)
    top_ctr  = np.array([[cx, cy, cz + half_h]], dtype=np.float32)

    # Vertex layout: [0]=bot_center, [1..n]=bot_ring, [n+1..2n]=top_ring, [2n+1]=top_center
    verts = np.vstack([bot_ctr, bot_ring, top_ring, top_ctr])

    faces = []
    # Bottom cap (winding: outward normal faces -Z)
    for j in range(n_segs):
        faces.append([0, 1 + (j + 1) % n_segs, 1 + j])
    # Side quads (two triangles each)
    for j in range(n_segs):
        b0 = 1 + j;               b1 = 1 + (j + 1) % n_segs
        t0 = n_segs + 1 + j;      t1 = n_segs + 1 + (j + 1) % n_segs
        faces.append([b0, b1, t0])
        faces.append([b1, t1, t0])
    # Top cap (winding: outward normal faces +Z)
    tc = 2 * n_segs + 1
    for j in range(n_segs):
        faces.append([tc, n_segs + 1 + j, n_segs + 1 + (j + 1) % n_segs])

    return verts, np.array(faces, dtype=np.uint32)


# ── Vispy visualizer ──────────────────────────────────────────────────────────

class CylinderYarnVispy:
    def __init__(self, sim: YarnCylinderSim, fps: int = 60):
        self.sim    = sim
        self.paused = False
        self.frame  = 0
        self._build_scene()
        self._timer = app.Timer(interval=1.0 / fps, connect=self._on_timer, start=True)

    def _build_scene(self):
        self.canvas = scene.SceneCanvas(
            title="Yarn + Cylinder Collision — NVIDIA Warp PBD (vispy)",
            size=(960, 720),
            bgcolor=BG_COL,
            keys="interactive",
            show=True,
        )
        self.canvas.events.key_press.connect(self._on_key)

        view = self.canvas.central_widget.add_view()
        view.camera = scene.cameras.TurntableCamera(
            elevation=20,
            azimuth=-60,
            distance=5.5,
            up="y",
            fov=45,
        )

        visuals.XYZAxis(parent=view.scene)

        # ── Cylinder mesh ─────────────────────────────────────────────────────
        verts, faces = _make_cylinder_mesh(
            CYL_CENTER_X, CYL_CENTER_Y, CYL_CENTER_Z,
            CYL_RADIUS, CYL_HALF_H,
        )
        cyl_mesh = visuals.Mesh(
            vertices=verts,
            faces=faces,
            color=CYL_COL,
            shading="smooth",
        )
        cyl_mesh.parent = view.scene

        # ── Yarn ──────────────────────────────────────────────────────────────
        p = self.sim.positions()
        self._yarn = visuals.Line(
            pos=p,
            color=YARN_COL,
            width=3,
            connect="strip",
            parent=view.scene,
        )

        # ── Anchor & free-end markers ─────────────────────────────────────────
        self._anchor = visuals.Markers(parent=view.scene)
        self._anchor.set_data(p[:1],  face_color=ANCHOR_COL, size=14, edge_width=0)

        self._free_end = visuals.Markers(parent=view.scene)
        self._free_end.set_data(p[-1:], face_color=FREE_COL,   size=14, edge_width=0)

        # ── HUD ───────────────────────────────────────────────────────────────
        self._hud = visuals.Text(
            text=self._hud_text(),
            color="white",
            font_size=10,
            anchor_x="left",
            anchor_y="top",
            parent=self.canvas.scene,
            pos=(10, 20),
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
            f"frame {self.frame:04d}  {status}  [{self.sim.device}]\n"
            "Space=pause  R=reset  drag=orbit  scroll=zoom"
        )

    def run(self):
        app.run()


# ── Main ──────────────────────────────────────────────────────────────────────

wp.init()
sim = YarnCylinderSim()
viz = CylinderYarnVispy(sim)
viz.run()
