"""visualization.py — Real-time 3-D matplotlib animation for the yarn simulation."""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from simulation import Simulation

# ── Colour palette ────────────────────────────────────────────────────────────
BG_COL     = "#1a1a2e"
YARN_COL   = "#e94560"
DOT_COL    = "#f5a623"
ANCHOR_COL = "#00d4ff"


class Visualizer:
    def __init__(self, sim: Simulation, fps: int = 60):
        self.sim = sim
        self.interval = int(1000 / fps)   # ms between frames
        self._build_figure()

    # ── Public API ────────────────────────────────────────────────────────────

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

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_figure(self):
        self.fig = plt.figure(figsize=(7, 8), facecolor=BG_COL)
        self.ax  = self.fig.add_subplot(111, projection="3d", facecolor=BG_COL)

        ax = self.ax
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 0.5)
        ax.set_zlim(-2.5, 2.5)
        ax.set_xlabel("X", color="white")
        ax.set_ylabel("Y", color="white")
        ax.set_zlabel("Z", color="white")
        ax.tick_params(colors="white")
        ax.set_title("Yarn Simulation — NVIDIA Warp PBD",
                     color="white", pad=14, fontsize=13)

        for spine in [ax.xaxis, ax.yaxis, ax.zaxis]:
            spine.label.set_color("white")
            spine._axinfo["tick"]["color"] = "white"

        p = self.sim.positions()

        (self._yarn,) = ax.plot(
            p[:, 0], p[:, 1], p[:, 2],
            "-o", color=YARN_COL, linewidth=2.0,
            markersize=2.5, markerfacecolor=DOT_COL, zorder=5,
        )
        ax.scatter(
            [p[0, 0]], [p[0, 1]], [p[0, 2]],
            color=ANCHOR_COL, s=100, zorder=10, label="Anchor",
        )
        ax.legend(loc="upper right", labelcolor="white",
                  facecolor="#0f3460", edgecolor="gray")

        self._info = ax.text2D(
            0.02, 0.96, "", transform=ax.transAxes,
            color="white", fontsize=9,
        )

    def _update(self, frame: int):
        self.sim.step()
        p = self.sim.positions()
        self._yarn.set_data_3d(p[:, 0], p[:, 1], p[:, 2])
        self._info.set_text(
            f"frame {frame:04d}  t={self.sim.time:.2f}s  [{self.sim.device}]"
        )
        return self._yarn, self._info
