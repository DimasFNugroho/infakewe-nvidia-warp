"""examples/yarn_freefall.py — Horizontal yarn with one clamped end, free-fall.

Differences from the default simulation
----------------------------------------
- No wind
- Yarn starts horizontal along the X axis
- Left end is clamped (fixed anchor)
- Right end and all other particles fall freely under gravity

Run
---
    python examples/yarn_freefall.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import polyscope as ps
import warp as wp

import config
from simulation import Simulation


# ── Custom simulation: horizontal layout, no wind ─────────────────────────────

class FreeFallYarn(Simulation):
    def _initial_arrays(self):
        """Horizontal layout along X axis."""
        n      = config.NUM_PARTICLES
        pos_np = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            pos_np[i] = [i * config.REST_LEN, 0.0, 0.0]
        vel_np = np.zeros((n, 3), dtype=np.float32)
        return pos_np, vel_np

    def _wind(self) -> wp.vec3:
        return wp.vec3(0.0, 0.0, 0.0)   # no wind


# ── Init ──────────────────────────────────────────────────────────────────────
wp.init()
sim = FreeFallYarn()

ps.init()
ps.set_up_dir("y_up")
ps.set_ground_plane_mode("shadow_only")
ps.set_background_color((0.07, 0.07, 0.14, 1.0))

# ── Register geometry ─────────────────────────────────────────────────────────
p     = sim.positions()
edges = np.array([[i, i + 1] for i in range(len(p) - 1)])

yarn   = ps.register_curve_network("yarn", p, edges, radius=0.012)
yarn.set_color((0.91, 0.27, 0.38))

anchor = ps.register_point_cloud("anchor", p[:1], radius=0.04)
anchor.set_color((0.0, 0.83, 1.0))

free_end = ps.register_point_cloud("free end", p[-1:], radius=0.04)
free_end.set_color((1.0, 0.65, 0.0))

# ── Callback ──────────────────────────────────────────────────────────────────
paused = [False]

def callback():
    import polyscope.imgui as imgui

    imgui.SetNextWindowPos((10, 10), imgui.ImGuiCond_Once)
    imgui.SetNextWindowSize((260, 160), imgui.ImGuiCond_Once)
    imgui.Begin("Free-Fall Yarn")

    imgui.Text(f"Device : {sim.device}")
    imgui.Text(f"Time   : {sim.time:.2f} s")
    imgui.TextColored((0.0, 0.83, 1.0, 1.0), "● Clamped end")
    imgui.TextColored((1.0, 0.65, 0.0, 1.0), "● Free end")
    imgui.Separator()

    if imgui.Button("Pause" if not paused[0] else "Resume"):
        paused[0] = not paused[0]

    imgui.SameLine()
    if imgui.Button("Reset"):
        sim.reset()

    imgui.End()

    if not paused[0]:
        sim.step()
        new_pos = sim.positions()
        yarn.update_node_positions(new_pos)
        anchor.update_point_positions(new_pos[:1])
        free_end.update_point_positions(new_pos[-1:])

ps.set_user_callback(callback)
ps.show()
