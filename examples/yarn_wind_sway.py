"""examples/yarn_polyscope.py — Yarn simulation with Polyscope 3D visualization.

Run
---
    pip install polyscope
    python examples/yarn_polyscope.py

Controls (Polyscope built-in)
-----------------------------
    Left drag    rotate camera
    Right drag   pan
    Scroll       zoom
    Space        pause / resume
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import polyscope as ps
import warp as wp

from simulation import Simulation

# ── Init ──────────────────────────────────────────────────────────────────────
wp.init()
sim = Simulation()

ps.init()
ps.set_up_dir("y_up")
ps.set_ground_plane_mode("shadow_only")
ps.set_background_color((0.07, 0.07, 0.14, 1.0))

# ── Register the yarn as a curve network ─────────────────────────────────────
p = sim.positions()                          # (N, 3)

# Edges: connect each particle to the next one
edges = np.array([[i, i + 1] for i in range(len(p) - 1)])

yarn = ps.register_curve_network("yarn", p, edges, radius=0.012)
yarn.set_color((0.91, 0.27, 0.38))          # #e94560

# Anchor marker
anchor = ps.register_point_cloud("anchor", p[:1], radius=0.04)
anchor.set_color((0.0, 0.83, 1.0))          # #00d4ff

# ── Simulation state ──────────────────────────────────────────────────────────
paused = [False]

def callback():
    import polyscope.imgui as imgui

    # ── GUI panel ─────────────────────────────────────────────────────────────
    imgui.SetNextWindowPos((10, 10), imgui.ImGuiCond_Once)
    imgui.SetNextWindowSize((260, 160), imgui.ImGuiCond_Once)
    imgui.Begin("Yarn Simulation")

    imgui.Text(f"Device  : {sim.device}")
    imgui.Text(f"Time    : {sim.time:.2f} s")
    imgui.Text(f"Nodes   : {sim.pos.shape[0]}")
    imgui.Separator()

    if imgui.Button("Pause" if not paused[0] else "Resume"):
        paused[0] = not paused[0]

    imgui.SameLine()
    if imgui.Button("Reset"):
        sim.reset()

    imgui.End()

    # ── Step + update geometry ────────────────────────────────────────────────
    if not paused[0]:
        sim.step()
        new_pos = sim.positions()
        yarn.update_node_positions(new_pos)
        anchor.update_point_positions(new_pos[:1])

ps.set_user_callback(callback)
ps.show()
