"""simulation.py — Yarn simulation state and per-frame step.

The Simulation class owns all Warp arrays and exposes a single
step() method that advances the yarn by one frame (config.DT seconds).
"""

import numpy as np
import warp as wp

import config
from kernels import (
    kernel_integrate,
    kernel_stretch_even,
    kernel_stretch_odd,
    kernel_bend,
    kernel_update_velocity,
)


class Simulation:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if wp.is_cuda_available() else "cpu")
        self.time   = 0.0

        # ── Pre-compute kernel launch dimensions ──────────────────────────────
        n = config.NUM_PARTICLES
        self._n_even = n // 2
        self._n_odd  = (n - 1) // 2
        self._n_bend = n - 2

        pos_np, vel_np = self._initial_arrays()
        inv_mass_np    = np.ones(n, dtype=np.float32)
        inv_mass_np[0] = 0.0   # pin the anchor (top particle)

        # ── Warp arrays ───────────────────────────────────────────────────────
        self.pos      = wp.array(pos_np,        dtype=wp.vec3, device=self.device)
        self.vel      = wp.array(vel_np,        dtype=wp.vec3, device=self.device)
        self.prev_pos = wp.array(pos_np.copy(), dtype=wp.vec3, device=self.device)
        self.inv_mass = wp.array(inv_mass_np,   dtype=float,   device=self.device)

    # ── Public API ────────────────────────────────────────────────────────────

    def step(self):
        """Advance the simulation by one frame (config.DT seconds)."""
        sub_dt  = config.DT / config.SUBSTEPS
        inv_sdt = 1.0 / sub_dt
        wind    = self._wind()

        for _ in range(config.SUBSTEPS):
            self._predict(wind, sub_dt)
            self._solve_constraints()
            self._correct_velocity(inv_sdt)

        self.time += config.DT

    def reset(self):
        """Restore the simulation to its initial state without reallocating arrays."""
        self.time = 0.0
        pos_np, vel_np = self._initial_arrays()
        self.pos.assign(wp.array(pos_np,        dtype=wp.vec3, device=self.device))
        self.prev_pos.assign(wp.array(pos_np.copy(), dtype=wp.vec3, device=self.device))
        self.vel.assign(wp.array(vel_np,        dtype=wp.vec3, device=self.device))

    def positions(self) -> np.ndarray:
        """Return current particle positions as a (N, 3) NumPy array."""
        return self.pos.numpy()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _initial_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (pos, vel) numpy arrays for the default initial state."""
        n      = config.NUM_PARTICLES
        pos_np = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            pos_np[i] = [0.0, -i * config.REST_LEN, 0.0]
        vel_np = np.zeros((n, 3), dtype=np.float32)
        return pos_np, vel_np

    def _wind(self) -> wp.vec3:
        t = self.time
        return wp.vec3(
            float(config.WIND_AMP_X * np.sin(2.0 * np.pi * config.WIND_FREQ * t)),
            0.0,
            float(config.WIND_AMP_Z * np.cos(2.0 * np.pi * config.WIND_FREQ * t * 0.7)),
        )

    def _predict(self, wind: wp.vec3, sub_dt: float):
        wp.launch(kernel_integrate, dim=config.NUM_PARTICLES, device=self.device,
                  inputs=[self.pos, self.vel, self.prev_pos, self.inv_mass,
                          config.GRAVITY, wind, sub_dt, config.DAMPING])

    def _solve_constraints(self):
        for _ in range(config.CONSTRAINT_ITER):
            wp.launch(kernel_stretch_even, dim=self._n_even, device=self.device,
                      inputs=[self.pos, self.inv_mass, config.REST_LEN, config.STRETCH_STIFF])
            wp.launch(kernel_stretch_odd,  dim=self._n_odd,  device=self.device,
                      inputs=[self.pos, self.inv_mass, config.REST_LEN, config.STRETCH_STIFF])
            wp.launch(kernel_bend,         dim=self._n_bend, device=self.device,
                      inputs=[self.pos, self.inv_mass, config.REST_LEN, config.BEND_STIFF])

    def _correct_velocity(self, inv_sdt: float):
        wp.launch(kernel_update_velocity, dim=config.NUM_PARTICLES, device=self.device,
                  inputs=[self.pos, self.prev_pos, self.vel, self.inv_mass, inv_sdt])
