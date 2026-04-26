"""algorithm3.py — Simulation step orchestrator (OGC Algorithm 3).

Mirrors the high-level structure of Algorithm 3 in the paper:

  1.  Compute the inertia target  Y = X_t + dt*v_t + dt^2 * a_ext
  2.  For each inner iteration:
        a.  (Optionally re-)run contact detection:
                Algorithm 1 → VF contacts
                Algorithm 2 → EE contacts
        b.  Run one inner solver iteration (Algorithm 4):
                stretch + bend + contact projection
  3.  Derive the new velocity from the position delta.

Simplifications vs. the paper
-----------------------------
* No BVH — brute-force detection (obstacle has O(100) tris).
* No conservative-bound truncation / re-detection trigger — contact detection
  runs every substep.
* PBD inner solver (matches the parent project) instead of VBD.

The public entry point is the `OGCSimulation` class, which is a drop-in
replacement for `simulation.Simulation` in the parent project.
"""

from __future__ import annotations

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

from .mesh import OGCMesh
from .algorithm1 import ObstacleGPU, VFContacts, detect_vertex_facet
from .algorithm2 import EEContacts, detect_edge_edge
from .algorithm4 import project_vf, project_ee, apply_vf_friction, apply_ee_friction, damp_normal_velocity, clamp_velocity


class OGCSimulation:
    """Yarn simulator with OGC contact against a static triangulated obstacle."""

    def __init__(
        self,
        obstacle_mesh: OGCMesh,
        contact_radius: float,
        device: str | None = None,
        contact_stiffness: float = 1.0,
    ):
        self.device = device or ("cuda" if wp.is_cuda_available() else "cpu")
        self.time   = 0.0

        self.r                  = float(contact_radius)
        self.contact_stiffness  = float(contact_stiffness)
        self.mu_static          = 0.4
        self.mu_kinetic         = 0.2
        self.v_max              = 20.0   # m/s; set to 0.0 to disable

        n = config.NUM_PARTICLES
        self._n_even = n // 2
        self._n_odd  = (n - 1) // 2
        self._n_bend = n - 2

        # ── Initial yarn state: straight line from origin toward end, left end pinned ──
        self.yarn_origin = np.zeros(3, dtype=np.float32)
        self.yarn_end    = np.array([config.YARN_LENGTH, 0.0, 0.0], dtype=np.float32)
        pos_np, vel_np = self._initial_arrays()
        self._particle_mass = 1.0
        inv_mass_np         = self._make_inv_mass(self._particle_mass, n)

        # ── Warp arrays for the yarn ──────────────────────────────────────────
        self.pos      = wp.array(pos_np,        dtype=wp.vec3, device=self.device)
        self.vel      = wp.array(vel_np,        dtype=wp.vec3, device=self.device)
        self.prev_pos = wp.array(pos_np.copy(), dtype=wp.vec3, device=self.device)
        self.inv_mass = wp.array(inv_mass_np,   dtype=float,   device=self.device)

        # Yarn edge list: consecutive segments (i, i+1) for i in 0..n-2
        edges_np = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1).astype(np.int32)
        self.yarn_edges = wp.array(edges_np, dtype=wp.vec2i, device=self.device)

        # ── Obstacle (static) — uploaded once ─────────────────────────────────
        self.obstacle = ObstacleGPU(obstacle_mesh, self.device)

        # ── Contact arrays (reused every step) ────────────────────────────────
        self.vf = VFContacts(n, self.device)
        self.ee = EEContacts(n - 1, self.device)

    # ── Public API ────────────────────────────────────────────────────────────

    def step(self):
        """One full time step = one call to Algorithm 3."""
        sub_dt  = config.DT / config.SUBSTEPS
        inv_sdt = 1.0 / sub_dt
        zero_wind = wp.vec3(0.0, 0.0, 0.0)

        for _ in range(config.SUBSTEPS):
            # Algorithm 3, line 3: predict → inertia target is baked into pos
            self._predict(zero_wind, sub_dt)

            # Algorithm 3, lines 5-14: (re)detect contacts
            detect_vertex_facet(self.pos, self.obstacle, self.vf, self.r, self.device)
            detect_edge_edge(
                self.pos, self.yarn_edges, self.obstacle, self.ee, self.r, self.device
            )

            # Algorithm 3, line 22: inner solver iterations (Algorithm 4).
            # Stretch, bend, and contact projection iterate to convergence.
            # Friction is applied ONCE after all iterations — it uses
            # prev_pos (substep start) as reference, so running it inside
            # the loop would compound the correction N times and over-constrain
            # tangential motion, causing instability at high constraint_iter.
            for _k in range(config.CONSTRAINT_ITER):
                self._stretch_pass()
                self._bend_pass()
                project_vf(self.pos, self.inv_mass, self.vf,
                           self.r, self.contact_stiffness, self.device)
                project_ee(self.pos, self.inv_mass, self.yarn_edges, self.ee,
                           self.r, self.contact_stiffness, self.device)

            # One friction pass per substep, using the converged position.
            apply_vf_friction(self.pos, self.prev_pos, self.inv_mass, self.vf,
                              self.r, self.mu_static, self.mu_kinetic, self.device)
            apply_ee_friction(self.pos, self.prev_pos, self.inv_mass, self.yarn_edges,
                              self.ee, self.r, self.mu_static, self.mu_kinetic, self.device)

            # Velocity update from position delta
            self._correct_velocity(inv_sdt)

        self.time += config.DT

    def reset(self):
        self.time = 0.0
        pos_np, vel_np = self._initial_arrays()
        self.pos.assign(wp.array(pos_np,        dtype=wp.vec3, device=self.device))
        self.prev_pos.assign(wp.array(pos_np.copy(), dtype=wp.vec3, device=self.device))
        self.vel.assign(wp.array(vel_np,        dtype=wp.vec3, device=self.device))

    def positions(self) -> np.ndarray:
        return self.pos.numpy()

    @property
    def particle_mass(self) -> float:
        return self._particle_mass

    @particle_mass.setter
    def particle_mass(self, value: float):
        value = max(float(value), 1e-6)   # guard against zero / negative
        if value == self._particle_mass:
            return
        self._particle_mass = value
        inv_np = self._make_inv_mass(value, config.NUM_PARTICLES)
        self.inv_mass.assign(wp.array(inv_np, dtype=float, device=self.device))

    # ── Private: PBD building blocks (same structure as simulation.py) ────────

    @staticmethod
    def _make_inv_mass(mass: float, n: int) -> np.ndarray:
        inv = np.full(n, 1.0 / mass, dtype=np.float32)
        inv[0] = 0.0   # particle 0 is always pinned (anchor)
        return inv

    def _initial_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        n = config.NUM_PARTICLES
        d = self.yarn_end - self.yarn_origin
        length = float(np.linalg.norm(d))
        d_hat = (d / length) if length > 1e-6 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        pos_np = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            pos_np[i] = self.yarn_origin + d_hat * (i * config.REST_LEN)
        return pos_np, np.zeros((n, 3), dtype=np.float32)

    def _predict(self, wind: wp.vec3, sub_dt: float):
        wp.launch(kernel_integrate, dim=config.NUM_PARTICLES, device=self.device,
                  inputs=[self.pos, self.vel, self.prev_pos, self.inv_mass,
                          config.GRAVITY, wind, sub_dt, config.DAMPING])

    def _stretch_pass(self):
        wp.launch(kernel_stretch_even, dim=self._n_even, device=self.device,
                  inputs=[self.pos, self.inv_mass, config.REST_LEN, config.STRETCH_STIFF])
        wp.launch(kernel_stretch_odd,  dim=self._n_odd,  device=self.device,
                  inputs=[self.pos, self.inv_mass, config.REST_LEN, config.STRETCH_STIFF])

    def _bend_pass(self):
        wp.launch(kernel_bend, dim=self._n_bend, device=self.device,
                  inputs=[self.pos, self.inv_mass, config.REST_LEN, config.BEND_STIFF])

    def _correct_velocity(self, inv_sdt: float):
        wp.launch(kernel_update_velocity, dim=config.NUM_PARTICLES, device=self.device,
                  inputs=[self.pos, self.prev_pos, self.vel, self.inv_mass, inv_sdt])
        damp_normal_velocity(self.vel, self.inv_mass, self.vf, self.r, self.device)
        clamp_velocity(self.vel, self.v_max, self.device)
