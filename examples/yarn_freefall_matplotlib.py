"""examples/yarn_freefall_matplotlib.py — Free-fall yarn visualised with matplotlib.

Same physics as yarn_freefall.py but rendered with matplotlib instead of polyscope.

Differences from the default simulation
----------------------------------------
- No wind
- Yarn starts horizontal along the X axis
- Left end is clamped (fixed anchor)
- Right end and all other particles fall freely under gravity

Run
---
    python examples/yarn_freefall_matplotlib.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import warp as wp

import config
from simulation import Simulation

# ── Colour palette ────────────────────────────────────────────────────────────
BG_COL     = "#12121f"
YARN_COL   = "#e94560"
DOT_COL    = "#f5a623"
ANCHOR_COL = "#00d4ff"
FREE_COL   = "#ffa500"


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


# ── Visualizer ────────────────────────────────────────────────────────────────

class FreeFallVisualizer:
    def __init__(self, sim: FreeFallYarn, fps: int = 60):
        self.sim      = sim
        self.interval = int(1000 / fps)
        self.paused   = False
        self._build_figure()

    def run(self):
        """Start the animation loop (blocks until the window is closed)."""
        self._ani = animation.FuncAnimation(
            self.fig,
            self._update,
            interval=self.interval,
            blit=False,
            cache_frame_data=False,
        )
        plt.tight_layout()
        plt.show()

    def _build_figure(self):
        self.fig = plt.figure(figsize=(9, 7), facecolor=BG_COL)
        self.ax  = self.fig.add_subplot(111, projection="3d", facecolor=BG_COL)

        ax = self.ax
        L  = config.YARN_LENGTH

        # Matplotlib's 3-D axes always treat Z as "up".  The simulation uses Y
        # as the gravity axis (gravity = -Y).  We therefore remap:
        #   plot X  ← sim X   (along the yarn)
        #   plot Y  ← sim Z   (depth — initially zero, small oscillations)
        #   plot Z  ← sim Y   (vertical in the window, falls downward)
        ax.set_xlim(-2.2, L + 0.2)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-L - 0.2, 0.5)
        ax.set_xlabel("X", color="white")
        ax.set_ylabel("Z (depth)", color="white")
        ax.set_zlabel("Y (gravity ↓)", color="white")
        ax.tick_params(colors="white")
        ax.set_title("Free-Fall Yarn — NVIDIA Warp PBD (matplotlib)",
                     color="white", pad=14, fontsize=12)

        # View from slightly above and to the side so the fall is clearly visible.
        ax.view_init(elev=15, azim=-70)

        for spine in [ax.xaxis, ax.yaxis, ax.zaxis]:
            spine.label.set_color("white")
            spine._axinfo["tick"]["color"] = "white"

        p = self.sim.positions()

        (self._yarn,) = ax.plot(
            p[:, 0], p[:, 2], p[:, 1],   # X, sim-Z→plotY, sim-Y→plotZ
            "-o", color=YARN_COL, linewidth=2.0,
            markersize=2.5, markerfacecolor=DOT_COL, zorder=5,
        )
        (self._anchor,) = ax.plot(
            [p[0, 0]], [p[0, 2]], [p[0, 1]],
            "o", color=ANCHOR_COL, markersize=10, zorder=10, label="Clamped end",
        )
        (self._free_end,) = ax.plot(
            [p[-1, 0]], [p[-1, 2]], [p[-1, 1]],
            "o", color=FREE_COL, markersize=10, zorder=10, label="Free end",
        )
        ax.legend(loc="upper right", labelcolor="white",
                  facecolor="#0f3460", edgecolor="gray")

        self._info = ax.text2D(
            0.02, 0.96, "", transform=ax.transAxes,
            color="white", fontsize=9,
        )

        # ── Pause on spacebar ─────────────────────────────────────────────────
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _on_key(self, event):
        if event.key == " ":
            self.paused = not self.paused
        elif event.key == "r":
            self.sim.reset()

    def _update(self, frame: int):
        if not self.paused:
            self.sim.step()

        p = self.sim.positions()
        self._yarn.set_data_3d(p[:, 0], p[:, 2], p[:, 1])
        self._anchor.set_data_3d([p[0, 0]], [p[0, 2]], [p[0, 1]])
        self._free_end.set_data_3d([p[-1, 0]], [p[-1, 2]], [p[-1, 1]])
        status = "PAUSED  (space=resume, r=reset)" if self.paused else f"t={self.sim.time:.2f}s"
        self._info.set_text(
            f"frame {frame:04d}  {status}  [{self.sim.device}]\n"
            "space=pause  r=reset"
        )
        return self._yarn, self._anchor, self._free_end, self._info


# ── Main ──────────────────────────────────────────────────────────────────────

wp.init()
sim = FreeFallYarn()
viz = FreeFallVisualizer(sim)
viz.run()
