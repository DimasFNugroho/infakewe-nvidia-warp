"""examples/yarn_plane_friction_vispy.py — Free-fall yarn on a tilted plane with friction.

Copy of yarn_cylinder_friction_vispy.py with the cylinder replaced by a
tilted rectangular plane.  Both ends of the yarn are free.

  High friction  → yarn grips the ramp and stays where it lands.
  Low  friction  → yarn slides down the slope and falls off the lower edge.

Tunable constants
-----------------
  TILT_DEG    inclination of the ramp in degrees (right/+X side is lower)
  MU_STATIC   static  friction coefficient
  MU_KINETIC  kinetic friction coefficient (must be ≤ MU_STATIC)

Controls
--------
  Space       — pause / resume
  R           — reset simulation
  Left-drag   — orbit camera
  Scroll      — zoom
  Middle-drag — pan

Run
---
    python examples/yarn_plane_friction_vispy.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import numpy as np
import warp as wp

from vispy import app, scene
from vispy.scene import visuals

import config
from simulation import Simulation


# ── Plane parameters ──────────────────────────────────────────────────────────
TILT_DEG     = 25.0          # tilt around Z; right (+X) side is lower
PLANE_CX     =  1.0
PLANE_CY     = -0.55
PLANE_CZ     =  0.0
PLANE_HALF_W =  3.0          # half-extent along the slope (u axis)
PLANE_HALF_D =  2.0          # half-extent along depth (Z axis)

_th      = math.radians(TILT_DEG)
PLANE_N  = wp.vec3( math.sin(_th),  math.cos(_th), 0.0)  # outward normal
PLANE_U  = wp.vec3( math.cos(_th), -math.sin(_th), 0.0)  # downhill direction
PLANE_V  = wp.vec3(0.0, 0.0, 1.0)                         # depth (Z)
PLANE_PT = wp.vec3(PLANE_CX, PLANE_CY, PLANE_CZ)

# ── Friction coefficients ─────────────────────────────────────────────────────
MU_STATIC  = 0.6
MU_KINETIC = 0.4

# ── Colours ───────────────────────────────────────────────────────────────────
BG_COL    = (0.07, 0.07, 0.13, 1.0)
YARN_COL  = (0.91, 0.27, 0.38)
FREE_COL  = (1.0,  0.65, 0.0)
PLANE_COL = (0.35, 0.55, 0.85, 0.80)


# ── Combined collision + friction kernel ──────────────────────────────────────

@wp.kernel
def kernel_plane_collision_friction(
    pos:      wp.array(dtype=wp.vec3),
    prev_pos: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    plane_pt: wp.vec3,
    plane_n:  wp.vec3,
    plane_u:  wp.vec3,
    plane_v:  wp.vec3,
    half_w:   float,
    half_d:   float,
    mu_s:     float,
    mu_k:     float,
):
    i = wp.tid()
    if inv_mass[i] == 0.0:
        return

    p   = pos[i]
    rel = p - plane_pt
    d   = wp.dot(rel, plane_n)   # signed distance; negative = below plane

    if d >= 0.0:
        return

    # Bounds check
    proj_rel = rel - d * plane_n
    if wp.abs(wp.dot(proj_rel, plane_u)) > half_w:
        return
    if wp.abs(wp.dot(proj_rel, plane_v)) > half_d:
        return

    depth = -d

    # Normal correction
    new_x = p[0] + depth * plane_n[0]
    new_y = p[1] + depth * plane_n[1]
    new_z = p[2] + depth * plane_n[2]

    # Tangential friction (identical logic to cylinder version)
    pp     = prev_pos[i]
    disp_x = new_x - pp[0]
    disp_y = new_y - pp[1]
    disp_z = new_z - pp[2]

    disp_dot_n = disp_x * plane_n[0] + disp_y * plane_n[1] + disp_z * plane_n[2]
    tan_x = disp_x - disp_dot_n * plane_n[0]
    tan_y = disp_y - disp_dot_n * plane_n[1]
    tan_z = disp_z - disp_dot_n * plane_n[2]

    tan_mag = wp.sqrt(tan_x * tan_x + tan_y * tan_y + tan_z * tan_z)

    if tan_mag > 1.0e-8:
        if tan_mag <= mu_s * depth:
            new_x -= tan_x
            new_y -= tan_y
            new_z -= tan_z
        else:
            keep   = mu_k * depth / tan_mag
            new_x -= tan_x * (1.0 - keep)
            new_y -= tan_y * (1.0 - keep)
            new_z -= tan_z * (1.0 - keep)

    pos[i] = wp.vec3(new_x, new_y, new_z)


# ── Contact damping kernel ────────────────────────────────────────────────────

@wp.kernel
def kernel_plane_contact_damping(
    pos:      wp.array(dtype=wp.vec3),
    vel:      wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    plane_pt: wp.vec3,
    plane_n:  wp.vec3,
    plane_u:  wp.vec3,
    plane_v:  wp.vec3,
    half_w:   float,
    half_d:   float,
    skin:     float,
):
    i = wp.tid()
    if inv_mass[i] == 0.0:
        return

    p   = pos[i]
    rel = p - plane_pt
    d   = wp.dot(rel, plane_n)

    if d < -skin or d > skin:
        return

    proj_rel = rel - d * plane_n
    if wp.abs(wp.dot(proj_rel, plane_u)) > half_w:
        return
    if wp.abs(wp.dot(proj_rel, plane_v)) > half_d:
        return

    v   = vel[i]
    v_n = wp.dot(v, plane_n)
    if v_n < 0.0:
        vel[i] = wp.vec3(v[0] - v_n * plane_n[0],
                         v[1] - v_n * plane_n[1],
                         v[2] - v_n * plane_n[2])


# ── Simulation ────────────────────────────────────────────────────────────────

class YarnPlaneFrictionSim(Simulation):
    def __init__(self, device=None):
        super().__init__(device)
        # Both ends free
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
        wp.launch(
            kernel_plane_collision_friction,
            dim=config.NUM_PARTICLES,
            device=self.device,
            inputs=[
                self.pos, self.prev_pos, self.inv_mass,
                PLANE_PT, PLANE_N, PLANE_U, PLANE_V,
                PLANE_HALF_W, PLANE_HALF_D,
                MU_STATIC, MU_KINETIC,
            ],
        )

    def _correct_velocity(self, inv_sdt: float):
        super()._correct_velocity(inv_sdt)
        wp.launch(
            kernel_plane_contact_damping,
            dim=config.NUM_PARTICLES,
            device=self.device,
            inputs=[
                self.pos, self.vel, self.inv_mass,
                PLANE_PT, PLANE_N, PLANE_U, PLANE_V,
                PLANE_HALF_W, PLANE_HALF_D,
                0.01,
            ],
        )


# ── Plane mesh (thin rectangular box) ────────────────────────────────────────

def _make_plane_mesh(thickness=0.04):
    """Thin box whose top face is the collision surface."""
    pt = np.array([PLANE_CX, PLANE_CY, PLANE_CZ], dtype=np.float32)
    n  = np.array([math.sin(_th),  math.cos(_th), 0.0], dtype=np.float32)
    u  = np.array([math.cos(_th), -math.sin(_th), 0.0], dtype=np.float32)
    v  = np.array([0.0, 0.0, 1.0],                      dtype=np.float32)

    hw, hd = PLANE_HALF_W, PLANE_HALF_D
    t = thickness / 2.0

    top    = [pt + n*t + s*u*hw + r*v*hd for s in (-1, 1) for r in (-1, 1)]
    bottom = [pt - n*t + s*u*hw + r*v*hd for s in (-1, 1) for r in (-1, 1)]
    # indices: [0]=(-u,-v), [1]=(-u,+v), [2]=(+u,-v), [3]=(+u,+v)  (top)
    #          [4]=(-u,-v), [5]=(-u,+v), [6]=(+u,-v), [7]=(+u,+v)  (bottom)

    verts = np.array(top + bottom, dtype=np.float32)

    faces = np.array([
        # top
        [0, 2, 3], [0, 3, 1],
        # bottom
        [4, 5, 7], [4, 7, 6],
        # sides
        [0, 1, 5], [0, 5, 4],
        [2, 6, 7], [2, 7, 3],
        [0, 4, 6], [0, 6, 2],
        [1, 3, 7], [1, 7, 5],
    ], dtype=np.uint32)

    return verts, faces


# ── Vispy visualizer ──────────────────────────────────────────────────────────

class PlaneYarnVispy:
    def __init__(self, sim: YarnPlaneFrictionSim, fps: int = 60):
        self.sim    = sim
        self.paused = False
        self.frame  = 0
        self._build_scene()
        self._timer = app.Timer(interval=1.0 / fps, connect=self._on_timer, start=True)

    def _build_scene(self):
        self.canvas = scene.SceneCanvas(
            title="Yarn + Inclined Plane Friction — NVIDIA Warp PBD (vispy)",
            size=(960, 720),
            bgcolor=BG_COL,
            keys="interactive",
            show=True,
        )
        self.canvas.events.key_press.connect(self._on_key)

        view = self.canvas.central_widget.add_view()
        view.camera = scene.cameras.TurntableCamera(
            elevation=20, azimuth=-60, distance=8.0, up="y", fov=45,
        )

        visuals.XYZAxis(parent=view.scene)

        verts, faces = _make_plane_mesh()
        visuals.Mesh(vertices=verts, faces=faces, color=PLANE_COL,
                     shading="flat", parent=view.scene)

        p = self.sim.positions()
        self._yarn = visuals.Line(pos=p, color=YARN_COL, width=3,
                                  connect="strip", parent=view.scene)

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
            f"tilt={TILT_DEG}deg  mu_s={MU_STATIC}  mu_k={MU_KINETIC}\n"
            "Space=pause  R=reset  drag=orbit  scroll=zoom"
        )

    def run(self):
        app.run()


# ── Main ──────────────────────────────────────────────────────────────────────

wp.init()
sim = YarnPlaneFrictionSim()
viz = PlaneYarnVispy(sim)
viz.run()
