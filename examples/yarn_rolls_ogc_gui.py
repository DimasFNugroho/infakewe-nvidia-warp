"""examples/yarn_rolls_ogc_gui.py — Roll-to-roll yarn transport with OGC contact.

A yarn is paid out from a feeding roll (A), deflected by a guide cylinder,
and wound onto a pulling roll (B).  All three cylinders are OGC obstacles.

    Particle 0    — kinematic, anchored to roll A surface (facing roll B).
    Particle N-1  — kinematic, rotates around roll B at the configured pull
                    speed, simulating yarn being wound up.
    Particles 1…N-2 — free, governed by gravity + PBD constraints + OGC contact.

Both roll positions and the guide position are adjustable at runtime via GUI
sliders; any change resets the simulation to a straight yarn between the two
new attachment points.

Running:
    python examples/yarn_rolls_ogc_gui.py

Prerequisites:
    sudo apt install -y python3-tk   # Linux only if tkinter is missing
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Shared parameter defaults ─────────────────────────────────────────────────

DEFAULTS = {
    # Yarn geometry
    "yarn_length":       50.02465483234714,   # metres — N is derived automatically from this and ogc_r
    "particle_density":  14.956803455723541,  # % of max stable particle count (10=coarse, 100=max density)
    # Physics
    "gravity_y":        -9.81,
    "particle_mass":     0.001,
    "damping":           0.9994074074074074,
    "stretch_stiff":     0.30303030303030304,
    "bend_stiff":        0.0,
    "substeps":          104,
    "constraint_iter":   1,
    # OGC contact
    "ogc_r":             0.005,
    "ogc_stiff":         1.0,
    "self_ee_stiff":     0.2938856015779093,  # stiffness for yarn self-collision projection
    "mu_static":         0.10907127429805616,
    "mu_kinetic":        0.011879049676025918,
    "v_max":             20.0,
    # Roll A — feeding roll
    "roll_a_x":         -0.8,
    "roll_a_y":          0.0,
    "roll_a_z":          0.0,
    "roll_a_radius":     0.15,
    "roll_a_mass":       0.5219330453563715,  # kg — sets rotational inertia of roll A
    "roll_a_bearing_damping": 0.9978401727861771,  # per-substep drag on omega  [0..1]
    "roll_a_torque_scale":    1.0,                 # gain on yarn→roll torque
    # Roll B — pulling roll
    "roll_b_x":          0.8,
    "roll_b_y":         -0.36363636363636376,
    "roll_b_z":          0.9090909090909092,
    "roll_b_radius":     0.15,
    "pull_speed":        5.0,   # m/s at roll B surface; negative = reverse
    # Tension sensors
    "sensor_offset":     5,     # particles away from guide contact midpoint on each side
    # Self-collision
    "self_collision":    1,     # 1 = yarn self-collision on, 0 = off
    "redetect_threshold": 0.3,  # fraction of r; re-run detection only when max displacement exceeds this
    # Visualisation
    "heatmap_mode":      0,     # 0 = stripe colours, 1 = stretch heatmap
    "heatmap_max_strain": 0.1099622030237581,
    # Guide cylinder
    "cyl_x":             0.0,
    "cyl_y":            -0.32444444444444454,
    "cyl_z":            -0.03111111111111109,
    "cyl_radius":        0.08,
}


# ── Simulation worker process ─────────────────────────────────────────────────

def sim_worker(cmd_queue, shared, script_dir: str, defaults: dict):
    """Run Warp + OGC (3 obstacles) + vispy in a dedicated process."""
    sys.path.insert(0, os.path.join(script_dir, ".."))
    sys.path.insert(0, script_dir)

    import queue as py_queue
    import time
    import numpy as np
    import warp as wp
    from vispy import app, scene

    from vispy.scene import visuals

    import config
    # ── Override yarn geometry for the roll-to-roll scenario ─────────────────
    config.YARN_LENGTH   = float(defaults.get("yarn_length", 7.0))
    _n_max = max(4, int(config.YARN_LENGTH / float(defaults.get("ogc_r", 0.005))))
    _pct   = max(0.0, min(100.0, float(defaults.get("particle_density", 30.0))))
    config.NUM_PARTICLES = max(4, round(4 + (_pct / 100.0) * (_n_max - 4)))
    config.REST_LEN      = config.YARN_LENGTH / (config.NUM_PARTICLES - 1)

    from ogc.mesh       import build_cylinder, mesh_for_render
    from ogc.algorithm1 import ObstacleGPU, VFContacts, detect_vertex_facet
    from ogc.algorithm2 import EEContacts, detect_edge_edge
    from ogc.algorithm4 import (project_vf, project_ee,
                                 apply_vf_friction, apply_ee_friction,
                                 damp_normal_velocity, clamp_velocity,
                                 roll_a_torque_step, roll_b_motor_step,
                                 set_particle)
    from ogc.algorithm5 import SelfEEContacts, detect_self_ee
    from ogc.algorithm6 import project_self_ee, apply_self_ee_friction
    from kernels import (kernel_integrate,
                         kernel_stretch_even, kernel_stretch_odd,
                         kernel_bend, kernel_update_velocity,
                         kernel_reset_scalar, kernel_max_disp_sq)

    # ── Scene / rendering constants ───────────────────────────────────────────
    CYL_HALF_H = 1.5
    CYL_N_SEGS = 48
    N          = config.NUM_PARTICLES   # reassigned by do_reinit on num_particles change

    BG_COL     = (0.07, 0.07, 0.13, 1.0)
    ROLL_A_COL = (0.20, 0.50, 0.90, 0.80)   # blue  — feeding roll
    ROLL_B_COL = (0.90, 0.45, 0.10, 0.80)   # orange — pulling roll
    CYL_COL    = (0.25, 0.70, 0.45, 0.75)   # green  — guide cylinder
    ANCHOR_COL = (0.0,  0.83, 1.0)           # cyan marker — particle 0
    PULL_COL   = (1.0,  0.65, 0.0)           # amber marker — particle N-1

    STRIPE_SIZE    = 5
    STRIPE_PALETTE = np.array([
        [0.95, 0.25, 0.35, 1.0],
        [0.98, 0.80, 0.12, 1.0],
        [0.18, 0.82, 0.95, 1.0],
        [0.65, 0.40, 0.95, 1.0],
    ], dtype=np.float32)

    def make_yarn_colors(n):
        idx = (np.arange(n) // STRIPE_SIZE) % len(STRIPE_PALETTE)
        return STRIPE_PALETTE[idx]

    def compute_stretch_colors(pos_np: np.ndarray) -> np.ndarray:
        """Per-particle RGBA heatmap: blue = relaxed, yellow = moderate, red = high stretch.

        Strain is normalised so that 5 % elongation maps to full red.  Each
        particle gets the average strain of its two adjacent segments (endpoints
        use only their single adjacent segment).
        """
        segs      = pos_np[1:] - pos_np[:-1]           # (N-1, 3)
        seg_lens  = np.linalg.norm(segs, axis=1)        # (N-1,)
        strain    = (seg_lens - config.REST_LEN) / config.REST_LEN  # (N-1,)

        n = len(pos_np)
        p_strain          = np.empty(n, dtype=np.float32)
        p_strain[0]       = strain[0]
        p_strain[1:-1]    = 0.5 * (strain[:-1] + strain[1:])
        p_strain[-1]      = strain[-1]

        max_strain = max(float(state.get("heatmap_max_strain", 0.05)), 1e-6)
        t = np.clip(p_strain / max_strain, 0.0, 1.0)

        # Blue (0,0,1) → Yellow (1,1,0) → Red (1,0,0)
        r = np.where(t < 0.5, 2.0 * t,       1.0      ).astype(np.float32)
        g = np.where(t < 0.5, 2.0 * t, 2.0 * (1.0 - t)).astype(np.float32)
        b = np.where(t < 0.5, 1.0 - 2.0 * t, 0.0     ).astype(np.float32)
        a = np.ones(n, dtype=np.float32)
        return np.stack([r, g, b, a], axis=1)

    # ── Mutable runtime state ─────────────────────────────────────────────────
    state    = dict(defaults)
    running  = [False]
    frame    = [0]
    sim_time = [0.0]
    angle_b  = [0.0]   # current winding angle of particle N-1 on roll B (rad)
    angle_a  = [0.0]   # current rotation angle of roll A (rad)
    omega_a  = [0.0]   # roll A angular velocity (rad/s)

    # ── Geometry helpers ──────────────────────────────────────────────────────

    def init_angle_a() -> float:
        """Initial Roll A angle: departure point faces Roll B."""
        ax, ay = state["roll_a_x"], state["roll_a_y"]
        bx, by = state["roll_b_x"], state["roll_b_y"]
        return float(np.arctan2(by - ay, bx - ax))

    def roll_b_attach(angle: float) -> np.ndarray:
        """Point on roll B surface at the given winding angle."""
        bx, by, bz = state["roll_b_x"], state["roll_b_y"], state["roll_b_z"]
        rb = float(state["roll_b_radius"])
        return np.array([bx + rb * np.cos(angle), by + rb * np.sin(angle), bz],
                        dtype=np.float32)

    def init_angle_b() -> float:
        """Starting angle on roll B pointing toward roll A."""
        ax, ay = state["roll_a_x"], state["roll_a_y"]
        bx, by = state["roll_b_x"], state["roll_b_y"]
        return float(np.arctan2(ay - by, ax - bx))

    def make_initial_positions() -> tuple:
        """Helical winding on Roll A, then a nearly-taut free span to Roll B.

        n_free is computed from the straight-line gap distance so that free-span
        particles start at roughly REST_LEN spacing (taut).  All remaining
        particles go to the wound section, giving more wraps and a denser
        winding appearance.

        Returns (positions_array, n_wound) so callers can track the wound count.
        """
        ax, ay, az = state["roll_a_x"], state["roll_a_y"], state["roll_a_z"]
        bx, by, bz = state["roll_b_x"], state["roll_b_y"], state["roll_b_z"]
        r         = float(state["ogc_r"])
        orbit_r_a = max(float(state["roll_a_radius"]), 1e-6) + r
        orbit_r_b = max(float(state["roll_b_radius"]), 1e-6) + r

        # How many free-span particles are needed to cover the gap at REST_LEN?
        center_dist = float(np.linalg.norm([bx - ax, by - ay, bz - az]))
        free_dist   = max(center_dist - orbit_r_a - orbit_r_b, config.REST_LEN)
        n_free  = max(2, min(int(round(free_dist / config.REST_LEN)), N - 3))
        n_wound = N - n_free

        # Angular step and Z drift per segment so adjacent wraps sit one yarn-diameter apart.
        dtheta = config.REST_LEN / orbit_r_a
        dz     = 2.0 * r * config.REST_LEN / (2.0 * np.pi * orbit_r_a)

        positions: list = []
        for i in range(n_wound):
            theta = angle_a[0] + i * dtheta
            positions.append([ax + orbit_r_a * np.cos(theta),
                               ay + orbit_r_a * np.sin(theta),
                               az + i * dz])

        # Free span: straight line from departure point to Roll B tip.
        p_dep = np.array(positions[-1], dtype=np.float32)
        p_end = np.array([bx + orbit_r_b * np.cos(angle_b[0]),
                          by + orbit_r_b * np.sin(angle_b[0]),
                          bz], dtype=np.float32)
        for i in range(1, n_free + 1):
            t = i / n_free
            positions.append(list(p_dep + t * (p_end - p_dep)))

        return np.array(positions[:N], dtype=np.float32), n_wound

    def make_inv_mass() -> np.ndarray:
        m = max(float(state["particle_mass"]), 1e-6)
        inv = np.full(N, 1.0 / m, dtype=np.float32)
        inv[0]  = 0.0   # particle 0  kinematic — anchored to roll A
        inv[-1] = 0.0   # particle N-1 kinematic — winding onto roll B
        return inv

    # ── Init Warp ─────────────────────────────────────────────────────────────
    wp.init()
    device = "cuda" if wp.is_cuda_available() else "cpu"

    angle_b[0] = init_angle_b()
    angle_a[0] = init_angle_a()

    # ── Build obstacle meshes + GPU objects ───────────────────────────────────
    # (angle_a_wp / omega_a_wp created after wp.init() below — post mesh setup)
    def _cyl(sx, sy, sz, sr):
        return build_cylinder(state[sx], state[sy], state[sz],
                              state[sr], CYL_HALF_H, n_segs=CYL_N_SEGS)

    mesh_a   = _cyl("roll_a_x", "roll_a_y", "roll_a_z", "roll_a_radius")
    mesh_b   = _cyl("roll_b_x", "roll_b_y", "roll_b_z", "roll_b_radius")
    mesh_mid = _cyl("cyl_x",    "cyl_y",    "cyl_z",    "cyl_radius")

    obs_a   = ObstacleGPU(mesh_a,   device)
    obs_b   = ObstacleGPU(mesh_b,   device)
    obs_mid = ObstacleGPU(mesh_mid, device)

    # ── Yarn GPU arrays ───────────────────────────────────────────────────────
    pos_np, n_wound = make_initial_positions()

    # Roll rotational state on GPU — updated each substep by their respective kernels.
    angle_a_wp = wp.array([angle_a[0]], dtype=float, device=device)
    omega_a_wp = wp.array([0.0],        dtype=float, device=device)
    angle_b_wp = wp.array([angle_b[0]], dtype=float, device=device)

    pos_wp      = wp.array(pos_np,                              dtype=wp.vec3, device=device)
    vel_wp      = wp.array(np.zeros((N, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    prev_pos_wp = wp.array(pos_np.copy(),                       dtype=wp.vec3, device=device)
    inv_mass_wp = wp.array(make_inv_mass(),                     dtype=float,   device=device)

    edges_np = np.stack([np.arange(N - 1), np.arange(1, N)], axis=1).astype(np.int32)
    yarn_edges_wp = wp.array(edges_np, dtype=wp.vec2i, device=device)

    n_even = N // 2
    n_odd  = (N - 1) // 2
    n_bend = N - 2

    # ── Contact arrays — one set per obstacle ─────────────────────────────────
    vf_a   = VFContacts(N, device);   ee_a   = EEContacts(N - 1, device)
    vf_b   = VFContacts(N, device);   ee_b   = EEContacts(N - 1, device)
    vf_mid = VFContacts(N, device);   ee_mid = EEContacts(N - 1, device)

    # Self-collision contact array (yarn vs. yarn).
    self_ee = SelfEEContacts(N - 1, device)

    # ── Conservative-bound redetection buffers (Phase 4) ─────────────────────
    pos_det_wp  = wp.zeros(N, dtype=wp.vec3, device=device)  # positions at last detection
    max_disp_buf = wp.zeros(1, dtype=float,   device=device)  # scalar accumulator
    _force_redetect = [True]   # True → skip threshold check, always detect

    # List-of-lists so individual obstacles can be hot-swapped.
    contacts = [
        [obs_a,   vf_a,   ee_a],
        [obs_b,   vf_b,   ee_b],
        [obs_mid, vf_mid, ee_mid],
    ]

    # ── Particle-count reinit ─────────────────────────────────────────────────
    # Called when the num_particles slider changes. Rebinds all N-dependent GPU
    # arrays via nonlocal so that every other closure sees the new arrays on its
    # next call — no changes required in _execute_substeps or sim_reset.

    def do_reinit(new_N: int):
        nonlocal N, n_even, n_odd, n_bend, n_wound
        nonlocal pos_wp, vel_wp, prev_pos_wp, inv_mass_wp, yarn_edges_wp
        nonlocal angle_a_wp, omega_a_wp, angle_b_wp
        nonlocal vf_a, ee_a, vf_b, ee_b, vf_mid, ee_mid, self_ee, contacts
        nonlocal yarn_colors, pos_det_wp, max_disp_buf

        N = max(4, new_N)
        config.NUM_PARTICLES = N
        config.REST_LEN      = config.YARN_LENGTH / (N - 1)
        n_even = N // 2
        n_odd  = (N - 1) // 2
        n_bend = N - 2

        angle_b[0] = init_angle_b()
        angle_a[0] = init_angle_a()
        angle_a_wp = wp.array([angle_a[0]], dtype=float, device=device)
        omega_a_wp = wp.array([0.0],        dtype=float, device=device)
        angle_b_wp = wp.array([angle_b[0]], dtype=float, device=device)

        pos_np, n_wound = make_initial_positions()
        pos_wp      = wp.array(pos_np,                               dtype=wp.vec3, device=device)
        vel_wp      = wp.array(np.zeros((N, 3), dtype=np.float32),   dtype=wp.vec3, device=device)
        prev_pos_wp = wp.array(pos_np.copy(),                        dtype=wp.vec3, device=device)
        inv_mass_wp = wp.array(make_inv_mass(),                      dtype=float,   device=device)

        edges_np      = np.stack([np.arange(N - 1), np.arange(1, N)], axis=1).astype(np.int32)
        yarn_edges_wp = wp.array(edges_np, dtype=wp.vec2i, device=device)

        vf_a   = VFContacts(N, device);   ee_a   = EEContacts(N - 1, device)
        vf_b   = VFContacts(N, device);   ee_b   = EEContacts(N - 1, device)
        vf_mid = VFContacts(N, device);   ee_mid = EEContacts(N - 1, device)
        self_ee = SelfEEContacts(N - 1, device)

        # Preserve existing obstacle GPU objects; only contact arrays change.
        contacts = [
            [contacts[0][0], vf_a,   ee_a],
            [contacts[1][0], vf_b,   ee_b],
            [contacts[2][0], vf_mid, ee_mid],
        ]

        pos_det_wp   = wp.zeros(N, dtype=wp.vec3, device=device)
        max_disp_buf = wp.zeros(1, dtype=float,   device=device)
        _force_redetect[0] = True

        yarn_colors = make_yarn_colors(N)
        _graph[0]   = None
        sim_time[0] = 0.0
        frame[0]    = 0
        print(f"[sim] reinit: N={N}  REST_LEN={config.REST_LEN:.5f}", flush=True)

    # ── CUDA graph capture state ──────────────────────────────────────────────
    # Replaying a captured graph eliminates Python kernel-launch overhead
    # (~16 000 launches/frame at current settings → ~80 ms saved/frame).
    # The graph is rebuilt whenever slider-controlled params change, with a
    # 300 ms debounce so continuous slider drag stays smooth.
    _graph        = [None]   # captured wp.Graph, or None if stale / first run
    _graph_params = [{}]     # scalar-param snapshot baked into _graph
    _last_snap    = [{}]     # snapshot from the previous frame
    _change_time  = [0.0]    # perf_counter() when params last differed from graph
    _frame_ms     = [0.0]    # wall time of the last sim_step() call (shown in HUD)
    _USE_GRAPH    = (device == "cuda")
    _DEBOUNCE     = 0.10     # seconds to wait for slider to settle before rebuild

    def apply_state():
        config.GRAVITY         = wp.vec3(0.0, float(state["gravity_y"]), 0.0)
        config.DAMPING         = float(state["damping"])
        config.STRETCH_STIFF   = float(state["stretch_stiff"])
        config.BEND_STIFF      = float(state["bend_stiff"])
        config.SUBSTEPS        = int(state["substeps"])
        config.CONSTRAINT_ITER = int(state["constraint_iter"])

    apply_state()

    def sim_reset():
        nonlocal n_wound
        angle_b[0] = init_angle_b()
        angle_a[0] = init_angle_a()
        omega_a[0] = 0.0
        angle_a_wp.assign(wp.array([angle_a[0]], dtype=float, device=device))
        omega_a_wp.assign(wp.array([0.0],        dtype=float, device=device))
        angle_b_wp.assign(wp.array([angle_b[0]], dtype=float, device=device))
        pos0, n_wound = make_initial_positions()
        pos_wp.assign(wp.array(pos0,                              dtype=wp.vec3, device=device))
        prev_pos_wp.assign(wp.array(pos0.copy(),                  dtype=wp.vec3, device=device))
        vel_wp.assign(wp.array(np.zeros((N, 3), dtype=np.float32), dtype=wp.vec3, device=device))
        inv_mass_wp.assign(wp.array(make_inv_mass(),               dtype=float,   device=device))
        sim_time[0] = 0.0
        frame[0]    = 0
        _force_redetect[0] = True

    # ── Shared substep body ───────────────────────────────────────────────────
    # Called both during CUDA graph capture (records launches without executing)
    # and in the Python-fallback path (executes launches immediately).

    def _snapshot_params() -> dict:
        """All scalar params that are baked into the CUDA graph at capture time."""
        return {
            "substeps":               config.SUBSTEPS,
            "constraint_iter":        config.CONSTRAINT_ITER,
            "gravity_y":              float(state["gravity_y"]),
            "damping":                float(state["damping"]),
            "stretch_stiff":          float(state["stretch_stiff"]),
            "bend_stiff":             float(state["bend_stiff"]),
            "ogc_r":                  float(state["ogc_r"]),
            "ogc_stiff":              float(state["ogc_stiff"]),
            "self_ee_stiff":          float(state.get("self_ee_stiff", 0.3)),
            "mu_static":              float(state["mu_static"]),
            "mu_kinetic":             float(state["mu_kinetic"]),
            "v_max":                  float(state["v_max"]),
            "pull_speed":             float(state["pull_speed"]),
            "particle_mass":          float(state["particle_mass"]),
            "roll_a_x":               float(state["roll_a_x"]),
            "roll_a_y":               float(state["roll_a_y"]),
            "roll_a_z":               float(state["roll_a_z"]),
            "roll_a_radius":          float(state["roll_a_radius"]),
            "roll_a_mass":            float(state["roll_a_mass"]),
            "roll_a_bearing_damping": float(state["roll_a_bearing_damping"]),
            "roll_a_torque_scale":    float(state["roll_a_torque_scale"]),
            "roll_b_x":               float(state["roll_b_x"]),
            "roll_b_y":               float(state["roll_b_y"]),
            "roll_b_z":               float(state["roll_b_z"]),
            "roll_b_radius":          float(state["roll_b_radius"]),
            "self_collision":         int(state.get("self_collision", 1)),
        }

    # ── Tension sensor state (sphere-window strategy) ────────────────────────
    # Two detection spheres are placed along the line from guide centre toward
    # each roll, just outside the guide contact zone. Any particles inside a
    # sphere are averaged for that sensor's tension reading.
    _sphere_centers = [np.zeros(3), np.zeros(3)]  # [upstream, downstream]

    def _sensor_sphere_centers():
        """Sphere centres: guide_centre + dir_to_roll × (cyl_radius + offset_m)."""
        gc  = np.array([float(state["cyl_x"]),
                        float(state["cyl_y"]),
                        float(state["cyl_z"])])
        ra  = np.array([float(state["roll_a_x"]),
                        float(state["roll_a_y"]),
                        float(state["roll_a_z"])])
        rb  = np.array([float(state["roll_b_x"]),
                        float(state["roll_b_y"]),
                        float(state["roll_b_z"])])
        offset_m = max(1, int(state.get("sensor_offset", 5))) * config.REST_LEN
        gap      = float(state["cyl_radius"]) + offset_m

        dir_a = ra - gc;  na = np.linalg.norm(dir_a)
        dir_b = rb - gc;  nb = np.linalg.norm(dir_b)
        if na > 1e-9: dir_a /= na
        if nb > 1e-9: dir_b /= nb
        return gc + dir_a * gap, gc + dir_b * gap

    def _tension_from_mask(pp, mask):
        """Average tension (cN) over all particles in boolean mask."""
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            return 0.0
        stiff      = float(state["stretch_stiff"])
        mass       = float(state["particle_mass"])
        L0         = config.REST_LEN
        sub_dt_ref = config.DT / 200.0   # normalise so reading is substep-stable
        exts = []
        for i in idxs:
            if i < N - 1:
                exts.append(max(0.0, np.linalg.norm(pp[i+1] - pp[i]) - L0))
            if i > 0:
                exts.append(max(0.0, np.linalg.norm(pp[i] - pp[i-1]) - L0))
        if not exts:
            return 0.0
        T_N = mass * stiff * float(np.mean(exts)) / (sub_dt_ref ** 2)
        return T_N * 100.0   # → centi-Newtons

    def _wrap_angle(pp, sc_a, sc_b):
        """Wrap angle θ (rad) between the two sensor sphere centres around guide."""
        cx = float(state["cyl_x"]);  cz = float(state["cyl_z"])
        va = np.array([sc_a[0] - cx, sc_a[2] - cz])
        vb = np.array([sc_b[0] - cx, sc_b[2] - cz])
        na = np.linalg.norm(va);  nb = np.linalg.norm(vb)
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.arccos(np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0)))

    def _write_shared(pp, sim_t):
        """Compute tension via detection spheres, then Capstan metrics → shared[]."""
        sc_a, sc_b = _sensor_sphere_centers()
        _sphere_centers[0] = sc_a
        _sphere_centers[1] = sc_b

        sphere_r = max(1, int(state.get("sensor_offset", 5))) * config.REST_LEN * 1.5
        dist_a   = np.linalg.norm(pp - sc_a, axis=1)
        dist_b   = np.linalg.norm(pp - sc_b, axis=1)
        mask_a   = dist_a < sphere_r
        mask_b   = dist_b < sphere_r

        T_a   = _tension_from_mask(pp, mask_a)
        T_b   = _tension_from_mask(pp, mask_b)
        theta = _wrap_angle(pp, sc_a, sc_b)
        mu_k  = float(state["mu_kinetic"])
        capstan_pred = T_a * np.exp(mu_k * theta)
        residual     = (T_b / capstan_pred) if capstan_pred > 1e-9 else 0.0
        # shared layout: [T_a, T_b, theta_deg, capstan_pred, residual, sim_time]
        shared[0] = T_a
        shared[1] = T_b
        shared[2] = float(np.degrees(theta))
        shared[3] = capstan_pred
        shared[4] = residual
        shared[5] = sim_t

    def _execute_substeps():
        """One full frame of substeps — graph-capture-safe.

        Every Python expression here is evaluated at the time this function
        is called.  In graph-capture mode that means the values are baked
        into the recorded graph nodes.  In Python-fallback mode they execute
        immediately with the current state values.
        """
        sub_dt        = config.DT / config.SUBSTEPS
        inv_sdt       = 1.0 / sub_dt
        r             = float(state["ogc_r"])
        stiff         = float(state["ogc_stiff"])
        self_ee_stiff = float(state.get("self_ee_stiff", 0.3))
        mu_s          = float(state["mu_static"])
        mu_k          = float(state["mu_kinetic"])
        v_max         = float(state["v_max"])
        self_coll_on  = bool(int(state.get("self_collision", 1)))
        zero_wind     = wp.vec3(0.0, 0.0, 0.0)

        ax, ay, az = float(state["roll_a_x"]), float(state["roll_a_y"]), float(state["roll_a_z"])
        ra         = max(float(state["roll_a_radius"]), 1e-6)
        M_a        = max(float(state["roll_a_mass"]), 1e-3)
        center_a   = wp.vec3(ax, ay, az)

        bx, by, bz = float(state["roll_b_x"]), float(state["roll_b_y"]), float(state["roll_b_z"])
        rb         = max(float(state["roll_b_radius"]), 1e-6)
        center_b   = wp.vec3(bx, by, bz)
        orbit_r_a  = ra + r
        orbit_r_b  = rb + r

        for _ in range(config.SUBSTEPS):
            roll_a_torque_step(
                pos_wp, center_a, ra, orbit_r_a,
                config.REST_LEN, config.STRETCH_STIFF,
                float(state["particle_mass"]), M_a, sub_dt,
                float(state["roll_a_bearing_damping"]),
                float(state["roll_a_torque_scale"]),
                200.0, angle_a_wp, omega_a_wp, device,
            )
            roll_b_motor_step(
                pos_wp, center_b, rb, orbit_r_b,
                float(state["pull_speed"]), sub_dt,
                N - 1, angle_b_wp, device,
            )
            wp.launch(kernel_integrate, dim=N, device=device,
                      inputs=[pos_wp, vel_wp, prev_pos_wp, inv_mass_wp,
                              config.GRAVITY, zero_wind, sub_dt, config.DAMPING])
            # Re-detect contacts every substep so projections use fresh contacts.
            # (Detection kernels are GPU-only and safe inside a CUDA graph.)
            for obs, vf, ee in contacts:
                detect_vertex_facet(pos_wp, obs, vf, r, device)
                detect_edge_edge(pos_wp, yarn_edges_wp, obs, ee, r, device)
            if self_coll_on:
                detect_self_ee(pos_wp, yarn_edges_wp, self_ee, r, device,
                               n_wound=n_wound)
            for _k in range(config.CONSTRAINT_ITER):
                wp.launch(kernel_stretch_even, dim=n_even, device=device,
                          inputs=[pos_wp, inv_mass_wp, config.REST_LEN, config.STRETCH_STIFF])
                wp.launch(kernel_stretch_odd,  dim=n_odd,  device=device,
                          inputs=[pos_wp, inv_mass_wp, config.REST_LEN, config.STRETCH_STIFF])
                wp.launch(kernel_bend,         dim=n_bend, device=device,
                          inputs=[pos_wp, inv_mass_wp, config.REST_LEN, config.BEND_STIFF])
                for obs, vf, ee in contacts:
                    project_vf(pos_wp, inv_mass_wp, vf, r, stiff, device)
                    project_ee(pos_wp, inv_mass_wp, yarn_edges_wp, ee, r, stiff, device)
                if self_coll_on:
                    project_self_ee(pos_wp, inv_mass_wp, yarn_edges_wp,
                                    self_ee, r, self_ee_stiff, device)
            # Roll A (contacts[0]): friction only on free-span particles — wound
            # particles follow the roll kinematically and must not be locked by
            # surface friction or the Capstan strain gradient is destroyed.
            obs_a_entry, vf_a_cur, ee_a_cur = contacts[0]
            apply_vf_friction(pos_wp, prev_pos_wp, inv_mass_wp,
                              vf_a_cur, r, mu_s, mu_k, device,
                              min_idx=n_wound)
            apply_ee_friction(pos_wp, prev_pos_wp, inv_mass_wp,
                              yarn_edges_wp, ee_a_cur, r, mu_s, mu_k, device,
                              min_idx=max(0, n_wound - 1))
            for obs, vf, ee in contacts[1:]:
                apply_vf_friction(pos_wp, prev_pos_wp, inv_mass_wp,
                                  vf, r, mu_s, mu_k, device)
                apply_ee_friction(pos_wp, prev_pos_wp, inv_mass_wp,
                                  yarn_edges_wp, ee, r, mu_s, mu_k, device)
            if self_coll_on:
                apply_self_ee_friction(pos_wp, prev_pos_wp, inv_mass_wp,
                                       yarn_edges_wp, self_ee, r, mu_s, mu_k, device)
            wp.launch(kernel_update_velocity, dim=N, device=device,
                      inputs=[pos_wp, prev_pos_wp, vel_wp, inv_mass_wp, inv_sdt])
            for obs, vf, ee in contacts:
                damp_normal_velocity(vel_wp, inv_mass_wp, vf, r, device)
            clamp_velocity(vel_wp, v_max, device)

    def _detect_contacts():
        """Seed contact arrays once (on init/reset) before the substep loop runs."""
        r = float(state["ogc_r"])
        self_coll = bool(int(state.get("self_collision", 1)))
        for obs, vf, ee in contacts:
            detect_vertex_facet(pos_wp, obs, vf, r, device)
            detect_edge_edge(pos_wp, yarn_edges_wp, obs, ee, r, device)
        if self_coll:
            detect_self_ee(pos_wp, yarn_edges_wp, self_ee, r, device, n_wound=n_wound)
        _force_redetect[0] = False

    def _rebuild_graph():
        """(Re)capture the substep loop as a CUDA graph.

        Must be called AFTER apply_state() so that config.* values are current.
        The capture does a dry-run (one non-executing pass) then stores the
        resulting graph for repeated replay.
        """
        apply_state()
        wp.capture_begin(device=device)
        _execute_substeps()
        _graph[0]        = wp.capture_end(device=device)
        _graph_params[0] = _snapshot_params()
        print(
            f"[sim] CUDA graph built: {config.SUBSTEPS} substeps × "
            f"{config.CONSTRAINT_ITER} iter",
            flush=True,
        )

    def sim_step():
        t0 = time.perf_counter()

        # Seed contact arrays on first frame after reset/reinit so the CUDA
        # graph has valid (non-empty) contact lists from the very first substep.
        if _force_redetect[0]:
            _detect_contacts()

        if _USE_GRAPH:
            snap = _snapshot_params()

            # Reset debounce timer whenever params differ from last frame.
            if snap != _last_snap[0]:
                _change_time[0] = t0
            _last_snap[0] = snap

            if _graph[0] is None or snap != _graph_params[0]:
                # Graph is stale or absent.
                # Rebuild immediately on first call; otherwise debounce so that
                # rapid slider drag keeps running (Python-loop fallback) without
                # triggering a rebuild on every frame.
                if _graph[0] is None or (t0 - _change_time[0]) >= _DEBOUNCE:
                    _rebuild_graph()
                    # fall through to capture_launch below
                else:
                    # Still within debounce window → run Python loop, correct params
                    apply_state()
                    _execute_substeps()
                    sim_time[0]   += config.DT
                    _frame_ms[0]   = (time.perf_counter() - t0) * 1000.0
                    return

            wp.capture_launch(_graph[0])

        else:
            # CPU fallback (no CUDA available) — always Python loop
            apply_state()
            _execute_substeps()

        sim_time[0]  += config.DT
        _frame_ms[0]  = (time.perf_counter() - t0) * 1000.0

    # ── Warm up Warp kernels ──────────────────────────────────────────────────
    print(f"[sim_worker] device={device} — warming up kernels...", flush=True)
    sim_step()
    sim_reset()
    print("[sim_worker] ready — GUI is live.", flush=True)

    # ── vispy scene ───────────────────────────────────────────────────────────
    canvas = scene.SceneCanvas(
        title="OGC roll-to-roll yarn (GUI-controlled)",
        size=(960, 720), bgcolor=BG_COL, keys="interactive", show=True,
    )
    view = canvas.central_widget.add_view()
    view.camera = scene.cameras.TurntableCamera(
        elevation=20, azimuth=-60, distance=5.5, up="y", fov=45,
    )
    visuals.XYZAxis(parent=view.scene)

    va, fa = mesh_for_render(mesh_a)
    vb, fb = mesh_for_render(mesh_b)
    vm, fm = mesh_for_render(mesh_mid)
    roll_a_vis = visuals.Mesh(vertices=va, faces=fa, color=ROLL_A_COL,
                              shading="smooth", parent=view.scene)
    roll_b_vis = visuals.Mesh(vertices=vb, faces=fb, color=ROLL_B_COL,
                              shading="smooth", parent=view.scene)
    cyl_vis    = visuals.Mesh(vertices=vm, faces=fm, color=CYL_COL,
                              shading="smooth", parent=view.scene)

    p = pos_wp.numpy()
    yarn_colors = make_yarn_colors(N)
    yarn_line   = visuals.Line(pos=p, color=yarn_colors, width=3,
                               connect="strip", parent=view.scene)
    marker_a = visuals.Markers(parent=view.scene)
    marker_b = visuals.Markers(parent=view.scene)
    marker_a.set_data(p[:1],  face_color=ANCHOR_COL, size=14, edge_width=0)
    marker_b.set_data(p[-1:], face_color=PULL_COL,   size=14, edge_width=0)

    # Tension sensor sphere markers: yellow = upstream (A), green = downstream (B)
    # Shown at sphere centre; larger size indicates the detection window.
    _sc_a0, _sc_b0 = _sensor_sphere_centers()
    _sphere_centers[0] = _sc_a0;  _sphere_centers[1] = _sc_b0
    sensor_marker_a = visuals.Markers(parent=view.scene)
    sensor_marker_b = visuals.Markers(parent=view.scene)
    sensor_marker_a.set_data(np.array([_sc_a0], dtype=np.float32),
                             face_color=(1.0, 0.9, 0.0, 0.8), size=16, edge_width=0)
    sensor_marker_b.set_data(np.array([_sc_b0], dtype=np.float32),
                             face_color=(0.0, 0.9, 0.2, 0.8), size=16, edge_width=0)

    # pos=(10, 60): text renders upward from this anchor in vispy canvas coords,
    # so 3 lines of font_size=10 need ~15px each → land at y≈60, 45, 30 (all visible).
    hud = visuals.Text(
        text="", color="white", font_size=10, anchor_x="left", anchor_y="top",
        parent=canvas.scene, pos=(10, 60),
    )

    # Warning banner shown while graph is rebuilding (params changed from UI).
    # Centered horizontally, near the top of the canvas.
    _warn = visuals.Text(
        text="", color=(1.0, 0.75, 0.0, 1.0), font_size=11,
        anchor_x="center", anchor_y="top",
        parent=canvas.scene, pos=(480, 40),
    )

    # ── Obstacle rebuild helpers ───────────────────────────────────────────────

    def rebuild_roll_a():
        new_mesh = _cyl("roll_a_x", "roll_a_y", "roll_a_z", "roll_a_radius")
        contacts[0][0] = ObstacleGPU(new_mesh, device)
        vv, ff = mesh_for_render(new_mesh)
        roll_a_vis.set_data(vertices=vv, faces=ff, color=ROLL_A_COL)
        _graph[0] = None   # obstacle GPU arrays changed — graph pointers stale
        sim_reset()

    def rebuild_roll_b():
        new_mesh = _cyl("roll_b_x", "roll_b_y", "roll_b_z", "roll_b_radius")
        contacts[1][0] = ObstacleGPU(new_mesh, device)
        vv, ff = mesh_for_render(new_mesh)
        roll_b_vis.set_data(vertices=vv, faces=ff, color=ROLL_B_COL)
        _graph[0] = None
        sim_reset()

    def rebuild_guide():
        new_mesh = _cyl("cyl_x", "cyl_y", "cyl_z", "cyl_radius")
        contacts[2][0] = ObstacleGPU(new_mesh, device)
        vv, ff = mesh_for_render(new_mesh)
        cyl_vis.set_data(vertices=vv, faces=ff, color=CYL_COL)
        _graph[0] = None

    # ── Main tick ─────────────────────────────────────────────────────────────
    def on_timer(_event):
        try:
            while True:
                cmd  = cmd_queue.get_nowait()
                kind = cmd[0]
                print(f"[sim_worker] cmd: {cmd}", flush=True)
                if   kind == "start": running[0] = True
                elif kind == "pause": running[0] = False
                elif kind == "reset":
                    running[0] = False
                    sim_reset()
                elif kind == "stop":
                    app.quit()
                    return
                elif kind == "param":
                    key, value = cmd[1], cmd[2]
                    state[key] = value

                    def _auto_reinit():
                        """Recompute N from yarn_length, ogc_r, and particle_density, then reinit."""
                        config.YARN_LENGTH = float(state["yarn_length"])
                        n_max = max(4, int(config.YARN_LENGTH / float(state["ogc_r"])))
                        pct   = max(0.0, min(100.0, float(state.get("particle_density", 30.0))))
                        new_N = max(4, round(4 + (pct / 100.0) * (n_max - 4)))
                        do_reinit(new_N)
                        yarn_line.set_data(pos=pos_wp.numpy(), color=yarn_colors,
                                           connect="strip")

                    if key in ("yarn_length", "ogc_r", "particle_density"):
                        _auto_reinit()
                    elif key in ("roll_a_x", "roll_a_y", "roll_a_z", "roll_a_radius"):
                        rebuild_roll_a()
                    elif key in ("roll_b_x", "roll_b_y", "roll_b_z", "roll_b_radius"):
                        rebuild_roll_b()
                    elif key in ("cyl_x", "cyl_y", "cyl_z", "cyl_radius"):
                        rebuild_guide()
                    elif key == "particle_mass":
                        inv_mass_wp.assign(
                            wp.array(make_inv_mass(), dtype=float, device=device)
                        )
                    apply_state()
        except py_queue.Empty:
            pass

        if running[0]:
            sim_step()
            frame[0] += 1

        pp = pos_wp.numpy()
        col = compute_stretch_colors(pp) if state.get("heatmap_mode") else yarn_colors
        yarn_line.set_data(pos=pp, color=col)
        marker_a.set_data(pp[:1],  face_color=ANCHOR_COL, size=14, edge_width=0)
        marker_b.set_data(pp[-1:], face_color=PULL_COL,   size=14, edge_width=0)

        _write_shared(pp, sim_time[0])
        sensor_marker_a.set_data(np.array([_sphere_centers[0]], dtype=np.float32),
                                 face_color=(1.0, 0.9, 0.0, 0.8), size=16, edge_width=0)
        sensor_marker_b.set_data(np.array([_sphere_centers[1]], dtype=np.float32),
                                 face_color=(0.0, 0.9, 0.2, 0.8), size=16, edge_width=0)

        # Read Roll A state from GPU once per frame (not per substep).
        _omega_a = float(omega_a_wp.numpy()[0])
        _angle_a = float(angle_a_wp.numpy()[0])

        status     = "RUN" if running[0] else "PAUSED"
        graph_mode = "graph" if (_USE_GRAPH and _graph[0] is not None
                                 and _snapshot_params() == _graph_params[0]) else "loop"
        _warn.text = (
            "-- Parameter changed from UI: simulation running slower until graph rebuilds --"
            if graph_mode == "loop" else ""
        )
        _T_a   = shared[0];  _T_b  = shared[1]
        _theta = shared[2];  _pred = shared[3];  _resid = shared[4]
        hud.text = (
            f"N={N}  seg={config.REST_LEN*1000:.1f}mm  r={state['ogc_r']:.3f}  "
            f"substeps={config.SUBSTEPS}  iter={config.CONSTRAINT_ITER}\n"
            f"pull={state['pull_speed']:+.2f} m/s  "
            f"ωA={_omega_a:+.1f} rad/s  θA={np.degrees(_angle_a):.0f}°\n"
            f"[{device}|{graph_mode}]  frame {frame[0]:05d}  {status}  "
            f"t={sim_time[0]:.2f}s  step={_frame_ms[0]:.1f}ms\n"
            f"T_A={_T_a:.2f}cN  T_B={_T_b:.2f}cN  "
            f"θ={_theta:.1f}°  pred={_pred:.2f}cN  resid={_resid:.3f}"
        )
        canvas.update()

    timer = app.Timer(interval=1.0 / 60.0, connect=on_timer, start=True)
    print("[sim_worker] entering vispy event loop", flush=True)
    app.run()
    del timer


# ── Tkinter control panel (parent process) ────────────────────────────────────

def run_ui(cmd_queue, shared):
    import json
    import tkinter as tk
    from tkinter import ttk, filedialog

    root = tk.Tk()
    root.title("OGC roll-to-roll yarn — controls")
    root.geometry("780x720")

    # ── Scrollable slider area ────────────────────────────────────────────────
    outer = ttk.Frame(root)
    outer.pack(fill="both", expand=True)

    canvas_tk  = tk.Canvas(outer, highlightthickness=0)
    scrollbar  = ttk.Scrollbar(outer, orient="vertical", command=canvas_tk.yview)
    scroll_frm = ttk.Frame(canvas_tk)

    scroll_frm.bind(
        "<Configure>",
        lambda e: canvas_tk.configure(scrollregion=canvas_tk.bbox("all")),
    )
    win_id = canvas_tk.create_window((0, 0), window=scroll_frm, anchor="nw")
    canvas_tk.configure(yscrollcommand=scrollbar.set)
    canvas_tk.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Keep scroll_frm width in sync with the canvas so sliders fill the window.
    canvas_tk.bind(
        "<Configure>",
        lambda e: canvas_tk.itemconfig(win_id, width=e.width),
    )

    # Mouse-wheel scrolling (Windows + Linux).
    canvas_tk.bind_all("<MouseWheel>",
                       lambda e: canvas_tk.yview_scroll(int(-1*(e.delta/120)), "units"))
    canvas_tk.bind_all("<Button-4>",
                       lambda e: canvas_tk.yview_scroll(-1, "units"))
    canvas_tk.bind_all("<Button-5>",
                       lambda e: canvas_tk.yview_scroll( 1, "units"))

    # Registries used by save / load.
    param_vars      = {}   # key → DoubleVar
    param_callbacks = {}   # key → on_change(v)

    def add_slider(label, key, from_, to_, default, is_int=False, fmt="{:.3f}"):
        frm = ttk.Frame(scroll_frm)
        frm.pack(fill="x", padx=8, pady=3)
        ttk.Label(frm, text=label, width=22).pack(side="left")

        val_var  = tk.DoubleVar(value=default)
        init_disp = f"{int(default)} / {int(to_)}" if is_int else fmt.format(default)
        disp_var = tk.StringVar(value=init_disp)
        ttk.Label(frm, textvariable=disp_var, width=10).pack(side="right")

        def on_change(v):
            if is_int:
                iv = int(round(float(v)))
                val_var.set(iv)
                disp_var.set(f"{iv} / {int(to_)}")
                cmd_queue.put(("param", key, iv))
            else:
                fv = float(v)
                val_var.set(fv)
                disp_var.set(fmt.format(fv))
                cmd_queue.put(("param", key, fv))

        ttk.Scale(frm, from_=from_, to=to_, variable=val_var,
                  orient="horizontal", command=on_change).pack(
            side="left", fill="x", expand=True, padx=6,
        )

        param_vars[key]      = val_var
        param_callbacks[key] = on_change

    def section(title):
        ttk.Label(scroll_frm, text=title, font=("", 10, "bold")).pack(
            anchor="w", padx=8, pady=(10, 0)
        )

    section("Visualisation")
    add_slider("Heatmap max strain", "heatmap_max_strain", 0.001, 1.0,
               DEFAULTS["heatmap_max_strain"], fmt="{:.3f}")

    section("Yarn geometry")
    _n_label_var = tk.StringVar(value="N = (computing...)")

    def _update_n_label():
        yl  = float(param_vars["yarn_length"].get())       if "yarn_length"       in param_vars else DEFAULTS["yarn_length"]
        ogr = float(param_vars["ogc_r"].get())             if "ogc_r"             in param_vars else DEFAULTS["ogc_r"]
        pct = float(param_vars["particle_density"].get())  if "particle_density"  in param_vars else DEFAULTS["particle_density"]
        n_max = max(4, int(yl / ogr))
        n     = max(4, round(4 + (pct / 100.0) * (n_max - 4)))
        _n_label_var.set(f"N = {n} particles  (max {n_max})")

    add_slider("Yarn length (m)", "yarn_length", 0.5, 60.0, DEFAULTS["yarn_length"],
               fmt="{:.2f}")
    param_vars["yarn_length"].trace_add("write", lambda *_: _update_n_label())
    add_slider("Particle density (%)", "particle_density", 10.0, 100.0,
               DEFAULTS["particle_density"], fmt="{:.1f}")
    param_vars["particle_density"].trace_add("write", lambda *_: _update_n_label())

    _n_frm = ttk.Frame(scroll_frm)
    _n_frm.pack(fill="x", padx=8, pady=2)
    ttk.Label(_n_frm, text="Particles (auto)", width=22).pack(side="left")
    ttk.Label(_n_frm, textvariable=_n_label_var, foreground="blue").pack(side="left")
    _update_n_label()  # populate immediately on startup

    section("Physics")
    add_slider("Gravity Y",          "gravity_y",      -20.0,  0.0,  DEFAULTS["gravity_y"],   fmt="{:+.2f}")
    add_slider("Particle mass (kg)", "particle_mass",    0.001, 10.0, DEFAULTS["particle_mass"], fmt="{:.4f}")
    add_slider("Damping",            "damping",          0.90,  1.00, DEFAULTS["damping"])
    add_slider("Stretch stiff",      "stretch_stiff",    0.0,   1.0,  DEFAULTS["stretch_stiff"])
    add_slider("Bend stiff",         "bend_stiff",       0.0,   1.0,  DEFAULTS["bend_stiff"])
    add_slider("Substeps",           "substeps",         1,    200,   DEFAULTS["substeps"],        is_int=True)
    add_slider("Constraint iter",    "constraint_iter",  1,     50,   DEFAULTS["constraint_iter"], is_int=True)

    section("OGC contact")
    add_slider("Contact radius r (m)", "ogc_r",        0.001, 0.20, DEFAULTS["ogc_r"],      fmt="{:.4f}")
    param_vars["ogc_r"].trace_add("write", lambda *_: _update_n_label())
    add_slider("Contact stiffness",    "ogc_stiff",    0.0,   1.0,  DEFAULTS["ogc_stiff"])
    add_slider("Friction μ_static",    "mu_static",    0.0,   1.0,  DEFAULTS["mu_static"])
    add_slider("Friction μ_kinetic",   "mu_kinetic",   0.0,   1.0,  DEFAULTS["mu_kinetic"])
    add_slider("Velocity max (m/s)",   "v_max",        0.0,  100.0, DEFAULTS["v_max"],       fmt="{:.1f}")

    section("Yarn self-collision")
    _self_coll_var = tk.IntVar(value=DEFAULTS["self_collision"])
    _self_coll_frm = ttk.Frame(scroll_frm)
    _self_coll_frm.pack(fill="x", padx=8, pady=3)
    def _on_self_coll():
        cmd_queue.put(("param", "self_collision", _self_coll_var.get()))
    ttk.Checkbutton(_self_coll_frm, text="Enable yarn self-collision",
                    variable=_self_coll_var, command=_on_self_coll).pack(side="left")
    param_vars["self_collision"]      = _self_coll_var
    param_callbacks["self_collision"] = lambda v: (
        _self_coll_var.set(int(v)), cmd_queue.put(("param", "self_collision", int(v)))
    )
    add_slider("Self-collision stiffness",   "self_ee_stiff",        0.0, 1.0,  DEFAULTS["self_ee_stiff"])
    add_slider("Redetect threshold (×r)",   "redetect_threshold",   0.05, 2.0, DEFAULTS["redetect_threshold"], fmt="{:.3f}")
    add_slider("Sensor offset (particles)", "sensor_offset",        1,   30,   DEFAULTS["sensor_offset"], is_int=True)

    section("Roll A — feeding roll (freely rotating)")
    add_slider("Roll A  X",       "roll_a_x",      -3.0,  3.0,  DEFAULTS["roll_a_x"],      fmt="{:+.3f}")
    add_slider("Roll A  Y",       "roll_a_y",      -3.0,  3.0,  DEFAULTS["roll_a_y"],      fmt="{:+.3f}")
    add_slider("Roll A  Z",       "roll_a_z",      -3.0,  3.0,  DEFAULTS["roll_a_z"],      fmt="{:+.3f}")
    add_slider("Roll A  radius",  "roll_a_radius",  0.02, 0.5,  DEFAULTS["roll_a_radius"])
    add_slider("Roll A  mass (kg)", "roll_a_mass",  0.01, 5.0,  DEFAULTS["roll_a_mass"])
    add_slider("Roll A  bearing damp", "roll_a_bearing_damping", 0.0, 1.0,
               DEFAULTS["roll_a_bearing_damping"])
    add_slider("Roll A  torque scale", "roll_a_torque_scale",    0.0, 1.0,
               DEFAULTS["roll_a_torque_scale"])

    section("Roll B — pulling roll")
    add_slider("Roll B  X",      "roll_b_x",      -3.0,  3.0, DEFAULTS["roll_b_x"],      fmt="{:+.3f}")
    add_slider("Roll B  Y",      "roll_b_y",      -3.0,  3.0, DEFAULTS["roll_b_y"],      fmt="{:+.3f}")
    add_slider("Roll B  Z",      "roll_b_z",      -3.0,  3.0, DEFAULTS["roll_b_z"],      fmt="{:+.3f}")
    add_slider("Roll B  radius", "roll_b_radius",  0.02, 0.5, DEFAULTS["roll_b_radius"])
    add_slider("Pull speed (m/s)", "pull_speed",  -5.0,  5.0, DEFAULTS["pull_speed"],    fmt="{:+.3f}")

    section("Guide cylinder")
    add_slider("Guide X",      "cyl_x",      -3.0,  3.0, DEFAULTS["cyl_x"],      fmt="{:+.3f}")
    add_slider("Guide Y",      "cyl_y",      -3.0,  3.0, DEFAULTS["cyl_y"],      fmt="{:+.3f}")
    add_slider("Guide Z",      "cyl_z",      -3.0,  3.0, DEFAULTS["cyl_z"],      fmt="{:+.3f}")
    add_slider("Guide radius", "cyl_radius",  0.02, 0.5, DEFAULTS["cyl_radius"])

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save_params():
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="params.json",
            title="Save parameters",
        )
        if not path:
            return
        data = {k: v.get() for k, v in param_vars.items()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[ui] saved parameters to {path}", flush=True)

    def load_params():
        path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Load parameters",
        )
        if not path:
            return
        with open(path) as f:
            data = json.load(f)
        for key, value in data.items():
            if key in param_vars and key in param_callbacks:
                param_vars[key].set(value)
                param_callbacks[key](value)
        # Apply DEFAULTS for any key missing from the file so that old param
        # files (saved before new sliders/toggles were added) still produce a
        # fully-defined state rather than silently leaving stale values.
        for key, default_val in DEFAULTS.items():
            if key not in data and key in param_vars and key in param_callbacks:
                param_vars[key].set(default_val)
                param_callbacks[key](default_val)
        _update_n_label()
        print(f"[ui] loaded parameters from {path}", flush=True)

    # ── Buttons (pinned to the bottom, outside the scroll area) ──────────────
    def send(cmd: str):
        print(f"[ui] send: {cmd}", flush=True)
        cmd_queue.put((cmd,))

    btn_frm = ttk.Frame(root)
    btn_frm.pack(fill="x", padx=8, pady=(4, 8), side="bottom")

    io_frm = ttk.Frame(root)
    io_frm.pack(fill="x", padx=8, pady=(0, 2), side="bottom")
    ttk.Button(io_frm, text="Save params", command=save_params
               ).pack(side="left", expand=True, fill="x", padx=2)
    ttk.Button(io_frm, text="Load params", command=load_params
               ).pack(side="left", expand=True, fill="x", padx=2)

    vis_frm = ttk.Frame(root)
    vis_frm.pack(fill="x", padx=8, pady=(0, 2), side="bottom")
    heatmap_var = tk.IntVar(value=DEFAULTS["heatmap_mode"])
    ttk.Checkbutton(
        vis_frm, text="Stretch heatmap  (blue = relaxed → yellow → red = high stretch)",
        variable=heatmap_var,
        command=lambda: cmd_queue.put(("param", "heatmap_mode", heatmap_var.get())),
    ).pack(side="left", padx=4)
    param_vars["heatmap_mode"]      = heatmap_var
    param_callbacks["heatmap_mode"] = lambda v: (
        heatmap_var.set(int(v)), cmd_queue.put(("param", "heatmap_mode", int(v)))
    )

    ttk.Button(btn_frm, text="Start", command=lambda: send("start")
               ).pack(side="left", expand=True, fill="x", padx=2)
    ttk.Button(btn_frm, text="Pause", command=lambda: send("pause")
               ).pack(side="left", expand=True, fill="x", padx=2)
    ttk.Button(btn_frm, text="Reset", command=lambda: send("reset")
               ).pack(side="left", expand=True, fill="x", padx=2)
    ttk.Button(btn_frm, text="Exit", command=lambda: on_close()
               ).pack(side="left", expand=True, fill="x", padx=2)

    # Auto-load params-main.json at startup so all saved preferences
    # (including toggles) are applied without the user needing to click Load.
    _autoload_path = os.path.join(_SCRIPT_DIR, "params-main.json")
    if os.path.exists(_autoload_path):
        try:
            with open(_autoload_path) as _f:
                _autoload_data = json.load(_f)
            for _key, _val in _autoload_data.items():
                if _key in param_vars and _key in param_callbacks:
                    param_vars[_key].set(_val)
                    param_callbacks[_key](_val)
            for _key, _default in DEFAULTS.items():
                if _key not in _autoload_data and _key in param_vars and _key in param_callbacks:
                    param_vars[_key].set(_default)
                    param_callbacks[_key](_default)
            _update_n_label()
            print(f"[ui] auto-loaded {_autoload_path}", flush=True)
        except Exception as _e:
            print(f"[ui] auto-load failed: {_e}", flush=True)

    # ── Live tension graph window ─────────────────────────────────────────────
    import collections, math
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    _GRAPH_HISTORY = 60      # seconds of rolling history to show
    _GRAPH_HZ      = 20      # update rate (Hz)
    _BUF_LEN       = _GRAPH_HISTORY * _GRAPH_HZ * 4   # ample buffer
    _t_buf  = collections.deque(maxlen=_BUF_LEN)
    _Ta_buf = collections.deque(maxlen=_BUF_LEN)
    _Tb_buf = collections.deque(maxlen=_BUF_LEN)

    graph_win = tk.Toplevel(root)
    graph_win.title("Tension sensors — Capstan analysis")
    graph_win.geometry("680x480")
    graph_win.protocol("WM_DELETE_WINDOW", lambda: None)   # keep alive with main

    fig, ax = plt.subplots(figsize=(6.8, 3.6), dpi=100)
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#1e1e1e")
    ax.tick_params(colors="white");  ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white");  ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555555")
    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("Tension (cN)")
    ax.set_title("Yarn tension — upstream (A) vs downstream (B)")
    ax.axhline(0, color="#555555", linewidth=0.8, linestyle="--")
    line_a, = ax.plot([], [], color="#4da6ff", linewidth=1.5, label="T_A upstream (yellow)")
    line_b, = ax.plot([], [], color="#ff7f3f", linewidth=1.5, label="T_B downstream (green)")
    ax.legend(facecolor="#2a2a2a", labelcolor="white", fontsize=8,
              loc="upper left")
    fig.tight_layout(rect=[0, 0.0, 1, 1])

    graph_canvas = FigureCanvasTkAgg(fig, master=graph_win)
    graph_canvas.get_tk_widget().pack(fill="both", expand=True)

    # Text panel below the plot for Capstan breakdown
    info_frm = ttk.Frame(graph_win)
    info_frm.pack(fill="x", padx=8, pady=4)
    _info_var = tk.StringVar(value="Waiting for simulation data...")
    ttk.Label(info_frm, textvariable=_info_var, font=("Courier", 9),
              foreground="white", background="#1e1e1e").pack(anchor="w")
    info_frm.configure(style="Dark.TFrame")

    _last_sim_t  = [-1.0]
    _graph_alive = [True]

    def _update_graph():
        if not _graph_alive[0]:
            return

        sim_t = shared[5]
        if sim_t != _last_sim_t[0]:
            _last_sim_t[0] = sim_t
            _t_buf.append(sim_t)
            _Ta_buf.append(shared[0])
            _Tb_buf.append(shared[1])

        if len(_t_buf) >= 2:
            t_arr  = list(_t_buf)
            ta_arr = list(_Ta_buf)
            tb_arr = list(_Tb_buf)
            t_now  = t_arr[-1]
            t_lo   = t_now - _GRAPH_HISTORY

            line_a.set_data(t_arr, ta_arr)
            line_b.set_data(t_arr, tb_arr)
            ax.set_xlim(max(0.0, t_lo), max(t_now, _GRAPH_HISTORY))
            all_vals = ta_arr + tb_arr
            v_min = min(all_vals);  v_max = max(all_vals)
            pad   = max(0.5, (v_max - v_min) * 0.15)
            ax.set_ylim(v_min - pad, v_max + pad)
            graph_canvas.draw_idle()

        T_a    = shared[0];  T_b   = shared[1]
        theta  = shared[2];  pred  = shared[3];  resid = shared[4]
        mu_k   = float(param_vars.get("mu_kinetic", tk.DoubleVar(value=0.0)).get())
        theta_r = math.radians(theta)
        _info_var.set(
            f"T_A = {T_a:7.3f} cN  (upstream,  yellow)      "
            f"T_B = {T_b:7.3f} cN  (downstream, green)\n"
            f"Wrap angle θ = {theta:.1f}°    μ_k = {mu_k:.4f}    "
            f"e^(μ_k·θ) = {math.exp(mu_k * theta_r):.4f}\n"
            f"Measured ratio T_B/T_A = {(T_b/T_a if T_a > 1e-6 else 0):.4f}    "
            f"Capstan pred T_B = {pred:.3f} cN    Residual = {resid:.4f}"
        )

        root.after(1000 // _GRAPH_HZ, _update_graph)

    root.after(500, _update_graph)   # start after half a second

    def on_close():
        _graph_alive[0] = False
        send("stop")
        root.after(150, root.destroy)
    root.protocol("WM_DELETE_WINDOW", on_close)

    root.mainloop()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # Shared memory: [T_a(cN), T_b(cN), theta_deg, capstan_pred(cN), residual, sim_time]
    shared = mp.Array('d', 6)

    cmd_queue = mp.Queue()
    worker = mp.Process(
        target=sim_worker, args=(cmd_queue, shared, _SCRIPT_DIR, DEFAULTS),
    )
    worker.start()

    try:
        run_ui(cmd_queue, shared)
    finally:
        cmd_queue.put(("stop",))
        worker.join(timeout=5.0)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=2.0)
        if worker.is_alive():
            worker.kill()   # SIGKILL — last resort if CUDA blocks SIGTERM
