"""examples/yarn_cylinder_friction_vispy.py — Free-fall yarn on a cylinder with surface friction.

Identical to yarn_cylinder_vispy.py but adds Coulomb friction between the
yarn particles and the cylinder surface.

Friction model (position-based, Müller et al. 2007)
----------------------------------------------------
During each collision constraint solve, the tangential displacement of a
contacting particle since the previous substep is measured and clamped:

  |Δpos_tangential| ≤ μ_s · penetration_depth   → static  (cancel slide)
  otherwise                                       → kinetic (reduce by μ_k · depth)

This is equivalent to applying a friction impulse proportional to the normal
correction — the standard Coulomb cone in position space.

Tunable constants
-----------------
  MU_STATIC   static  friction coefficient  (higher = stickier)
  MU_KINETIC  kinetic friction coefficient  (must be ≤ MU_STATIC)

Controls
--------
  Space       — pause / resume
  R           — reset simulation
  Left-drag   — orbit camera
  Scroll      — zoom
  Middle-drag — pan

Run
---
    python examples/yarn_cylinder_friction_vispy.py
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
CYL_CENTER_X =  1.0
CYL_CENTER_Y = -0.55
CYL_CENTER_Z =  0.0
CYL_RADIUS   =  0.22
CYL_HALF_H   =  1.5

# ── Friction coefficients ─────────────────────────────────────────────────────
MU_STATIC  = 1.9 #0.6   # try 0.0 (frictionless) → 1.0 (very rough)
MU_KINETIC = 0.09   # must be ≤ MU_STATIC

# ── Colours ───────────────────────────────────────────────────────────────────
BG_COL     = (0.07, 0.07, 0.13, 1.0)
YARN_COL   = (0.91, 0.27, 0.38)
ANCHOR_COL = (0.0,  0.83, 1.0)
FREE_COL   = (1.0,  0.65, 0.0)
CYL_COL    = (0.25, 0.70, 0.45, 0.75)


# ── Combined collision + friction kernel ──────────────────────────────────────

@wp.kernel
def kernel_cylinder_collision_friction(
    pos:        wp.array(dtype=wp.vec3),
    prev_pos:   wp.array(dtype=wp.vec3),
    inv_mass:   wp.array(dtype=float),
    cyl_center: wp.vec3,
    radius:     float,
    half_h:     float,
    mu_s:       float,
    mu_k:       float,
):
    """Resolve cylinder penetration and apply Coulomb friction in one pass.

    Normal correction:
        Push the particle radially outward onto the cylinder surface.
        The correction magnitude (penetration depth) is the normal impulse proxy.

    Tangential friction (position-based Coulomb):
        Measure how far the particle slid along the surface since prev_pos.
        Static  cone: |slide| ≤ mu_s * depth  →  cancel slide entirely.
        Kinetic cone: otherwise               →  reduce slide by mu_k * depth.
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
    if r >= radius:
        return

    # Degenerate: particle on the axis
    if r < 1.0e-8:
        pos[i] = wp.vec3(cyl_center[0] + radius, p[1], p[2])
        return

    # Outward surface normal
    nx = dx / r
    ny = dy / r

    # Penetration depth (proxy for normal constraint correction)
    depth = radius - r

    # ── Normal correction: push to surface ────────────────────────────────────
    new_x = cyl_center[0] + nx * radius
    new_y = cyl_center[1] + ny * radius
    new_z = p[2]

    # ── Tangential friction ───────────────────────────────────────────────────
    # Displacement since the last substep (prediction step moved pos forward;
    # prev_pos is where the particle was before prediction).
    pp     = prev_pos[i]
    disp_x = new_x - pp[0]
    disp_y = new_y - pp[1]
    disp_z = new_z - pp[2]   # Z is purely tangential (axial direction)

    # Remove the normal component to isolate the tangential (surface) slide
    disp_dot_n = disp_x * nx + disp_y * ny
    tan_x = disp_x - disp_dot_n * nx
    tan_y = disp_y - disp_dot_n * ny
    tan_z = disp_z                        # Z has no normal component

    tan_mag = wp.sqrt(tan_x * tan_x + tan_y * tan_y + tan_z * tan_z)

    if tan_mag > 1.0e-8:
        if tan_mag <= mu_s * depth:
            # Static friction: fully cancel the slide
            new_x -= tan_x
            new_y -= tan_y
            new_z -= tan_z
        else:
            # Kinetic friction: reduce slide, keeping mu_k * depth of it
            keep  = mu_k * depth / tan_mag
            new_x -= tan_x * (1.0 - keep)
            new_y -= tan_y * (1.0 - keep)
            new_z -= tan_z * (1.0 - keep)

    pos[i] = wp.vec3(new_x, new_y, new_z)


# ── Contact damping kernel (unchanged from base example) ─────────────────────

@wp.kernel
def kernel_cylinder_contact_damping(
    pos:        wp.array(dtype=wp.vec3),
    vel:        wp.array(dtype=wp.vec3),
    inv_mass:   wp.array(dtype=float),
    cyl_center: wp.vec3,
    radius:     float,
    half_h:     float,
    skin:       float,
):
    """Zero the inward radial velocity for particles resting on the cylinder."""
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
    if r > radius + skin or r < 1.0e-8:
        return

    nx    = dx / r
    ny    = dy / r
    v     = vel[i]
    v_rad = v[0] * nx + v[1] * ny
    if v_rad < 0.0:
        vel[i] = wp.vec3(v[0] - v_rad * nx,
                         v[1] - v_rad * ny,
                         v[2])


# ── Simulation ────────────────────────────────────────────────────────────────

class YarnCylinderFrictionSim(Simulation):
    def __init__(self, device=None):
        super().__init__(device)
        # Both ends are free — override the anchor pin set by the parent
        inv_mass_np = np.ones(config.NUM_PARTICLES, dtype=np.float32)
        self.inv_mass = wp.array(inv_mass_np, dtype=float, device=self.device)

    def _initial_arrays(self):
        n      = config.NUM_PARTICLES
        pos_np = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            pos_np[i] = [i * config.REST_LEN, 0.0, 0.0]
        return pos_np, np.zeros((n, 3), dtype=np.float32)

    def _wind(self) -> wp.vec3:
        return wp.vec3(0.0, 0.0, 0.0)

    def _solve_constraints(self):
        super()._solve_constraints()
        cyl = wp.vec3(CYL_CENTER_X, CYL_CENTER_Y, CYL_CENTER_Z)
        wp.launch(
            kernel_cylinder_collision_friction,
            dim=config.NUM_PARTICLES,
            device=self.device,
            inputs=[
                self.pos, self.prev_pos, self.inv_mass,
                cyl, CYL_RADIUS, CYL_HALF_H,
                MU_STATIC, MU_KINETIC,
            ],
        )

    def _correct_velocity(self, inv_sdt: float):
        super()._correct_velocity(inv_sdt)
        cyl = wp.vec3(CYL_CENTER_X, CYL_CENTER_Y, CYL_CENTER_Z)
        wp.launch(
            kernel_cylinder_contact_damping,
            dim=config.NUM_PARTICLES,
            device=self.device,
            inputs=[
                self.pos, self.vel, self.inv_mass,
                cyl, CYL_RADIUS, CYL_HALF_H,
                0.01,
            ],
        )


# ── Cylinder mesh ─────────────────────────────────────────────────────────────

def _make_cylinder_mesh(cx, cy, cz, radius, half_h, n_segs=48):
    angles  = np.linspace(0.0, 2.0 * np.pi, n_segs, endpoint=False, dtype=np.float32)
    cos_a, sin_a = np.cos(angles), np.sin(angles)

    bot_ring = np.column_stack([cx + radius * cos_a, cy + radius * sin_a,
                                np.full(n_segs, cz - half_h, dtype=np.float32)])
    top_ring = np.column_stack([cx + radius * cos_a, cy + radius * sin_a,
                                np.full(n_segs, cz + half_h, dtype=np.float32)])
    bot_ctr  = np.array([[cx, cy, cz - half_h]], dtype=np.float32)
    top_ctr  = np.array([[cx, cy, cz + half_h]], dtype=np.float32)
    verts    = np.vstack([bot_ctr, bot_ring, top_ring, top_ctr])

    faces = []
    for j in range(n_segs):
        faces.append([0, 1 + (j + 1) % n_segs, 1 + j])
    for j in range(n_segs):
        b0 = 1 + j;           b1 = 1 + (j + 1) % n_segs
        t0 = n_segs + 1 + j;  t1 = n_segs + 1 + (j + 1) % n_segs
        faces.append([b0, b1, t0])
        faces.append([b1, t1, t0])
    tc = 2 * n_segs + 1
    for j in range(n_segs):
        faces.append([tc, n_segs + 1 + j, n_segs + 1 + (j + 1) % n_segs])

    return verts, np.array(faces, dtype=np.uint32)


# ── Vispy visualizer ──────────────────────────────────────────────────────────

class CylinderFrictionVispy:
    def __init__(self, sim: YarnCylinderFrictionSim, fps: int = 60):
        self.sim    = sim
        self.paused = False
        self.frame  = 0
        self._build_scene()
        self._timer = app.Timer(interval=1.0 / fps, connect=self._on_timer, start=True)

    def _build_scene(self):
        self.canvas = scene.SceneCanvas(
            title="Yarn + Cylinder Friction — NVIDIA Warp PBD (vispy)",
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

        verts, faces = _make_cylinder_mesh(
            CYL_CENTER_X, CYL_CENTER_Y, CYL_CENTER_Z, CYL_RADIUS, CYL_HALF_H,
        )
        visuals.Mesh(vertices=verts, faces=faces, color=CYL_COL,
                     shading="smooth", parent=view.scene)

        p = self.sim.positions()
        self._yarn = visuals.Line(pos=p, color=YARN_COL, width=3,
                                  connect="strip", parent=view.scene)

        self._ends = visuals.Markers(parent=view.scene)
        self._ends.set_data(np.array([p[0], p[-1]]),
                            face_color=FREE_COL, size=14, edge_width=0)

        self._hud = visuals.Text(
            text=self._hud_text(), color="white", font_size=10,
            anchor_x="left", anchor_y="top",
            parent=self.canvas.scene, pos=(10, 20),
        )

    def _on_timer(self, _event):
        if not self.paused:
            self.sim.step()
            self.frame += 1

        p = self.sim.positions()
        self._yarn.set_data(pos=p)
        self._ends.set_data(np.array([p[0], p[-1]]),
                            face_color=FREE_COL, size=14, edge_width=0)
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
            f"frame {self.frame:04d}  {status}  [{self.sim.device}]   "
            f"mu_s={MU_STATIC}  mu_k={MU_KINETIC}\n"
            "Space=pause  R=reset  drag=orbit  scroll=zoom"
        )

    def run(self):
        app.run()


# ── Main ──────────────────────────────────────────────────────────────────────

wp.init()
sim = YarnCylinderFrictionSim()
viz = CylinderFrictionVispy(sim)
viz.run()
