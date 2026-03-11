"""examples/yarn_freefall_vispy.py — Free-fall yarn visualised with Vispy.

Same physics as yarn_freefall.py but rendered with Vispy (GPU-accelerated
OpenGL) instead of polyscope or matplotlib.

Differences from the default simulation
----------------------------------------
- No wind
- Yarn starts horizontal along the X axis
- Left end is clamped (fixed anchor)
- Right end and all other particles fall freely under gravity

Controls
--------
  Space       — pause / resume
  R           — reset simulation
  Left-drag   — orbit camera
  Scroll      — zoom
  Middle-drag — pan

Run
---
    python examples/yarn_freefall_vispy.py
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


# ── Colour palette ────────────────────────────────────────────────────────────
BG_COL     = (0.07, 0.07, 0.13, 1.0)
YARN_COL   = (0.91, 0.27, 0.38)
ANCHOR_COL = (0.0,  0.83, 1.0)
FREE_COL   = (1.0,  0.65, 0.0)


# ── Custom simulation: horizontal layout, no wind ─────────────────────────────

class FreeFallYarn(Simulation):
    def _initial_arrays(self):
        n      = config.NUM_PARTICLES
        pos_np = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            pos_np[i] = [i * config.REST_LEN, 0.0, 0.0]
        vel_np = np.zeros((n, 3), dtype=np.float32)
        return pos_np, vel_np

    def _wind(self) -> wp.vec3:
        return wp.vec3(0.0, 0.0, 0.0)


# ── Vispy visualizer ──────────────────────────────────────────────────────────

class FreeFallVispy:
    def __init__(self, sim: FreeFallYarn, fps: int = 60):
        self.sim    = sim
        self.paused = False
        self.frame  = 0
        self._build_scene()
        self._timer = app.Timer(interval=1.0 / fps, connect=self._on_timer, start=True)

    # ── Scene setup ───────────────────────────────────────────────────────────

    def _build_scene(self):
        self.canvas = scene.SceneCanvas(
            title="Free-Fall Yarn — NVIDIA Warp PBD (vispy)",
            size=(900, 700),
            bgcolor=BG_COL,
            keys="interactive",
            show=True,
        )
        self.canvas.events.key_press.connect(self._on_key)

        view = self.canvas.central_widget.add_view()
        view.camera = scene.cameras.TurntableCamera(
            elevation=15,
            azimuth=-70,
            distance=5.0,
            up="y",              # Y is gravity axis → keep Y as "up" in camera
            fov=45,
        )

        # ── Axis helper ───────────────────────────────────────────────────────
        visuals.XYZAxis(parent=view.scene)

        # ── Yarn line ─────────────────────────────────────────────────────────
        p = self.sim.positions()
        self._yarn = visuals.Line(
            pos=p,
            color=YARN_COL,
            width=3,
            connect="strip",
            parent=view.scene,
        )

        # ── Anchor marker ─────────────────────────────────────────────────────
        self._anchor = visuals.Markers(parent=view.scene)
        self._anchor.set_data(
            p[:1],
            face_color=ANCHOR_COL,
            size=14,
            edge_width=0,
        )

        # ── Free-end marker ───────────────────────────────────────────────────
        self._free_end = visuals.Markers(parent=view.scene)
        self._free_end.set_data(
            p[-1:],
            face_color=FREE_COL,
            size=14,
            edge_width=0,
        )

        # ── HUD text ──────────────────────────────────────────────────────────
        self._hud = visuals.Text(
            text=self._hud_text(),
            color="white",
            font_size=10,
            anchor_x="left",
            anchor_y="top",
            parent=self.canvas.scene,
            pos=(10, 20),
        )

        self._view = view

    # ── Timer callback (one frame) ────────────────────────────────────────────

    def _on_timer(self, _event):
        if not self.paused:
            self.sim.step()
            self.frame += 1

        p = self.sim.positions()
        self._yarn.set_data(pos=p)
        self._anchor.set_data(p[:1],  face_color=ANCHOR_COL, size=14, edge_width=0)
        self._free_end.set_data(p[-1:], face_color=FREE_COL,   size=14, edge_width=0)
        self._hud.text = self._hud_text()
        self.canvas.update()

    # ── Keyboard ──────────────────────────────────────────────────────────────

    def _on_key(self, event):
        if event.key == "Space":
            self.paused = not self.paused
        elif event.key == "R":
            self.sim.reset()
            self.frame = 0

    # ── Helpers ───────────────────────────────────────────────────────────────

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
sim = FreeFallYarn()
viz = FreeFallVispy(sim)
viz.run()
