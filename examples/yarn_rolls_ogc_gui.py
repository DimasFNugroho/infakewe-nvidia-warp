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

# ── Wrap-angle debug shared buffer ───────────────────────────────────────────
# Layout (all doubles):
#   [0]  n_search       — particles in search region (count, capped at MAX)
#   [1]  cx             — cylinder centre X (world)
#   [2]  cy             — cylinder centre Y (world)
#   [3]  r_cyl          — cylinder radius
#   [4]  r_contact      — r_cyl + ogc_r
#   [5]  arc_tol        — arc-band half-thickness
#   [6]  rd             — search radius
#   [7]  theta_rad      — computed wrap angle (rad)
#   [8]  n_arc          — count of arc particles
#   [9]  longest_len    — length of longest contiguous run
#   [10] refined_in_x   — refined in-tangent point (relative to cyl centre), NaN if none
#   [11] refined_in_y
#   [12] refined_out_x  — refined out-tangent point
#   [13] refined_out_y
#   [14] sim_t          — sim time when written
#   [15] signed_theta   — signed Σ Δθ (rad) before abs (drives sweep direction)
#   [16..23] reserved
# Per-particle body, starting at offset 24, stride 4:
#   [+0] x (relative to cyl centre)
#   [+1] y
#   [+2] r (distance to axis)
#   [+3] cls (0=search-only, 1=in-longest-run, 2=arc)
MAX_DEBUG_PARTS = 512
DBG_HEADER_LEN  = 24
DBG_STRIDE      = 4
DBG_ARRAY_LEN   = DBG_HEADER_LEN + DBG_STRIDE * MAX_DEBUG_PARTS


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
    # Roll A — tension servo (PI on T_a, overrides passive flywheel when on)
    "roll_a_servo_on":         0,     # 0 = passive flywheel, 1 = servo
    "roll_a_tension_setpoint": 5.0,   # target upstream tension (cN)
    "roll_a_kp":               1.0,   # proportional gain (rad/s per cN)
    "roll_a_ki":               0.1,   # integral gain (rad/s per cN·s)
    # Roll B — pulling roll
    "roll_b_x":          0.8,
    "roll_b_y":         -0.36363636363636376,
    "roll_b_z":          0.9090909090909092,
    "roll_b_radius":     0.15,
    "pull_speed":        5.0,   # m/s at roll B surface; negative = reverse
    # Tension sensor A (upstream, yellow) — place on Roll-A side of guide
    "sensor_a_x":  -0.30,  "sensor_a_y":  0.10,  "sensor_a_z":  0.00,
    # Detection volume = axis-aligned box.  Half-extents along each axis (m).
    # Defaults form an upright plate: 10 cm × 10 cm × 1 cm (thin in Z), oriented
    # so the yarn travelling in the XY plane passes broadside through the plate.
    "sensor_a_hx":  0.15,
    "sensor_a_hy":  0.15,
    "sensor_a_hz":  0.15,
    "sensor_a_alpha": 0.40,        # box visualization opacity, 0..1
    "sensor_a_cyl_r":    0.03,   # physical frictionless cylinder radius
    "sensor_a_cyl_enabled": 0,
    # Tension sensor B (downstream, green) — place on Roll-B side of guide
    "sensor_b_x":   0.20,  "sensor_b_y": -0.10,  "sensor_b_z":  0.20,
    "sensor_b_hx":  0.15,
    "sensor_b_hy":  0.15,
    "sensor_b_hz":  0.15,
    "sensor_b_alpha": 0.40,
    "sensor_b_cyl_r":    0.03,   # physical frictionless cylinder radius
    "sensor_b_cyl_enabled": 0,
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
    "cyl_detect_r":      0.20,   # detection radius for wrap-angle estimation (m)
    "cyl_detect_show":   0,      # 0 = hide detection volume, 1 = show
    # Initial warp state
}


# ── Simulation worker process ─────────────────────────────────────────────────

def sim_worker(cmd_queue, shared, dbg_shared, script_dir: str, defaults: dict):
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
                                 roll_a_torque_step, roll_a_servo_step,
                                 roll_b_motor_step, set_particle)
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
    CYL_DET_COL = (0.50, 0.70, 0.95, 0.15)   # light blue, very translucent — wrap-detect volume
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
    _servo_integral  = [0.0]   # PI integral term (cN·s)
    _servo_omega_cmd = [0.0]   # last computed servo omega command (rad/s, for HUD)
    _servo_t_prev    = [None]  # perf_counter() at previous PI tick

    # ── Geometry helpers ──────────────────────────────────────────────────────

    # ── Warp geometry helper ──────────────────────────────────────────────────
    def _external_tangent_points(C1, r1, C2, r2):
        """External tangent point pairs for two 2-D orbit circles.

        Returns up to two (T1, T2) pairs — T1 on circle-1, T2 on circle-2 —
        each at the foot of the perpendicular from its centre to the common
        external tangent line.  Returns [] when one circle encloses the other.
        """
        d = float(np.linalg.norm(C2 - C1))
        if d < 1e-9:
            return []
        cos_phi = (r2 - r1) / d
        if abs(cos_phi) > 1.0:
            return []
        phi = np.arccos(np.clip(cos_phi, -1.0, 1.0))
        u = (C2 - C1) / d
        pairs = []
        for sign in (+1.0, -1.0):
            c, s = np.cos(sign * phi), np.sin(sign * phi)
            n = np.array([u[0]*c - u[1]*s, u[0]*s + u[1]*c])
            pairs.append((C1 - r1 * n, C2 - r2 * n))
        return pairs

    def init_angle_a() -> float:
        """Initial Roll A angle: departure point faces guide cylinder."""
        ax, ay = state["roll_a_x"], state["roll_a_y"]
        cx, cy = state["cyl_x"],    state["cyl_y"]
        return float(np.arctan2(cy - ay, cx - ax))

    def roll_b_attach(angle: float) -> np.ndarray:
        """Point on roll B surface at the given winding angle."""
        bx, by, bz = state["roll_b_x"], state["roll_b_y"], state["roll_b_z"]
        rb = float(state["roll_b_radius"])
        return np.array([bx + rb * np.cos(angle), by + rb * np.sin(angle), bz],
                        dtype=np.float32)

    def init_angle_b() -> float:
        """Starting angle on roll B: surface point facing guide cylinder."""
        cx, cy = state["cyl_x"],    state["cyl_y"]
        bx, by = state["roll_b_x"], state["roll_b_y"]
        return float(np.arctan2(cy - by, cx - bx))

    def _warp_keypoints() -> dict:
        """Single source of truth for the warp geometry of the current configuration.

        Warp angle is computed purely from Roll A / guide / Roll B positions —
        no user-supplied angle parameter.  The perpendicular-to-span criterion
        selects the incoming and outgoing tangent sides consistent with the
        wrap direction (CCW or CW) determined by the A-C-B cross product.

        Returns a dict with 2-D XY numpy arrays and scalars:
          T_a_dep, T_c_in, T_c_out, T_b_arr  — orbit-surface tangent points
          theta_in, theta_out                 — angles on the guide orbit circle
          wrap_dir                            — +1 CCW, -1 CW
          warp_rad                            — warp angle in radians (geometric)
          orbit_r_a/c/b                       — orbit radii (phys radius + ogc_r)
        """
        ax, ay = float(state["roll_a_x"]), float(state["roll_a_y"])
        cx, cy = float(state["cyl_x"]),    float(state["cyl_y"])
        bx, by = float(state["roll_b_x"]), float(state["roll_b_y"])
        r      = float(state["ogc_r"])
        orbit_r_a = max(float(state["roll_a_radius"]), 1e-6) + r
        orbit_r_c = max(float(state["cyl_radius"]),    1e-6) + r
        orbit_r_b = max(float(state["roll_b_radius"]), 1e-6) + r

        C_a = np.array([ax, ay]); C_c = np.array([cx, cy]); C_b = np.array([bx, by])

        cross_z  = (ax - cx) * (by - cy) - (ay - cy) * (bx - cx)
        wrap_dir = +1 if cross_z < 0.0 else -1

        # Unit vectors and their 90°-CCW perpendiculars for each span
        def _unit(v):
            n = float(np.linalg.norm(v))
            return v / n if n > 1e-9 else np.array([1.0, 0.0])
        def _perp(u): return np.array([-u[1], u[0]])

        u_ac = _unit(C_c - C_a);  perp_ac = _perp(u_ac)
        u_cb = _unit(C_b - C_c);  perp_cb = _perp(u_cb)

        tans_ac = _external_tangent_points(C_a, orbit_r_a, C_c, orbit_r_c)
        tans_cb = _external_tangent_points(C_c, orbit_r_c, C_b, orbit_r_b)

        # Incoming tangent: pick T_c_in on the -wrap_dir side of the A→C perp
        # (yarn enters guide on the side opposite to the wrap-rotation direction)
        if tans_ac:
            T_a_dep, T_c_in = max(tans_ac,
                key=lambda p: -wrap_dir * float(np.dot(p[1] - C_c, perp_ac)))
        else:
            T_a_dep = C_a + orbit_r_a * u_ac
            T_c_in  = C_c - orbit_r_c * u_ac

        # Outgoing tangent: pick T_c_out on the -wrap_dir side of the C→B perp,
        # matching the incoming criterion so entry and exit lie on the same side
        # of the guide (opposite sign caused the chord to cut through the cylinder).
        if tans_cb:
            T_c_out, T_b_arr = max(tans_cb,
                key=lambda p: -wrap_dir * float(np.dot(p[0] - C_c, perp_cb)))
        else:
            T_c_out = C_c + orbit_r_c * u_cb
            T_b_arr = C_b - orbit_r_b * u_cb

        theta_in  = float(np.arctan2(T_c_in[1]  - cy, T_c_in[0]  - cx))
        theta_out = float(np.arctan2(T_c_out[1] - cy, T_c_out[0] - cx))
        warp_rad  = (wrap_dir * (theta_out - theta_in)) % (2.0 * np.pi)

        return dict(T_a_dep=T_a_dep, T_c_in=T_c_in, theta_in=theta_in,
                    theta_out=theta_out, T_c_out=T_c_out, T_b_arr=T_b_arr,
                    wrap_dir=wrap_dir, warp_rad=warp_rad,
                    orbit_r_a=orbit_r_a, orbit_r_c=orbit_r_c, orbit_r_b=orbit_r_b)

    def _auto_place_sensors():
        """Auto-place sensor A/B centers at span midpoints.

        Sensor A → midpoint of the free span from the last wound particle on
                   Roll A to the guide tangent-in point.
        Sensor B → midpoint of the guide tangent-out → Roll B arrival span.

        Uses the ACTUAL last wound particle position (not T_a_dep) to account
        for the angular advance of n_wound winding steps on Roll A.
        The Z offset corrects for helical Z-drift along the roll axis.

        Half-extents are NOT touched — user controls box size via sliders.
        Only position (x, y, z) and _sphere_centers are updated.
        """
        kp = _warp_keypoints()
        ax = float(state["roll_a_x"]);  ay = float(state["roll_a_y"])
        az = float(state["roll_a_z"])
        cz = float(state["cyl_z"])
        bz = float(state["roll_b_z"])
        orbit_r_a = kp["orbit_r_a"]

        # Actual last wound particle position — the yarn departs from here, not
        # from T_a_dep (which sits at angle_a[0] before any winding advance).
        dtheta = config.REST_LEN / orbit_r_a
        dz_per = 2.0 * float(state["ogc_r"]) * config.REST_LEN / (2.0 * np.pi * orbit_r_a)
        last_theta = angle_a[0] + (n_wound - 1) * dtheta
        dep_x = ax + orbit_r_a * np.cos(last_theta)
        dep_y = ay + orbit_r_a * np.sin(last_theta)
        dep_z = az + (n_wound - 1) * dz_per

        T_a_dep_3d = np.array([dep_x,              dep_y,              dep_z])
        T_c_in_3d  = np.array([kp["T_c_in"][0],  kp["T_c_in"][1],  cz])
        T_c_out_3d = np.array([kp["T_c_out"][0], kp["T_c_out"][1], cz])
        T_b_arr_3d = np.array([kp["T_b_arr"][0], kp["T_b_arr"][1], bz])

        mid_a = 0.5 * (T_a_dep_3d + T_c_in_3d)
        mid_b = 0.5 * (T_c_out_3d + T_b_arr_3d)

        # Orientation: local-X along the yarn tangent (so a thin-X box becomes a
        # plate perpendicular to the yarn). Local-Z is built from world Z and
        # then made orthogonal to X; local-Y completes the right-handed frame.
        def _frame(p0, p1):
            u = p1 - p0
            n = float(np.linalg.norm(u))
            if n < 1e-9:
                return np.eye(3, dtype=np.float32)
            ux = (u / n).astype(np.float32)
            ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            if abs(float(np.dot(ux, ref))) > 0.99:
                ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            uy = np.cross(ref, ux);  uy /= float(np.linalg.norm(uy))
            uz = np.cross(ux, uy)
            return np.column_stack([ux, uy.astype(np.float32), uz.astype(np.float32)])

        _sensor_R[0] = _frame(T_a_dep_3d, T_c_in_3d)
        _sensor_R[1] = _frame(T_c_out_3d, T_b_arr_3d)

        # Only update position — user controls half-extents via sliders.
        for k, v in (("sensor_a_x", mid_a[0]), ("sensor_a_y", mid_a[1]),
                     ("sensor_a_z", mid_a[2])):
            state[k] = float(v)
        _sphere_centers[0] = mid_a.copy()

        for k, v in (("sensor_b_x", mid_b[0]), ("sensor_b_y", mid_b[1]),
                     ("sensor_b_z", mid_b[2])):
            state[k] = float(v)
        _sphere_centers[1] = mid_b.copy()

    def make_initial_positions() -> tuple:
        """Auto-warp initial state: wound on Roll A → straight → arc on guide → straight → Roll B.

        Calls _warp_keypoints() for the tangent geometry, then distributes N
        particles across: wound coil, free span A→guide, arc on guide, free span guide→B.
        Also sets angle_a[0] and angle_b[0] to the exact tangent departure/arrival
        angles so that the GPU kinematic kernels start in sync.

        Returns (positions_array, n_wound).
        """
        kp = _warp_keypoints()

        ax, ay, az = float(state["roll_a_x"]), float(state["roll_a_y"]), float(state["roll_a_z"])
        cx, cy, cz = float(state["cyl_x"]),    float(state["cyl_y"]),    float(state["cyl_z"])
        bx, by, bz = float(state["roll_b_x"]), float(state["roll_b_y"]), float(state["roll_b_z"])
        r         = float(state["ogc_r"])
        orbit_r_a = kp["orbit_r_a"]
        orbit_r_c = kp["orbit_r_c"]

        T_a_dep   = kp["T_a_dep"];   T_c_in   = kp["T_c_in"]
        theta_in  = kp["theta_in"];  theta_out = kp["theta_out"]
        T_c_out   = kp["T_c_out"];   T_b_arr  = kp["T_b_arr"]
        wrap_dir  = kp["wrap_dir"];  warp_rad = kp["warp_rad"]

        # ── Update kinematic angles so GPU kernels start in sync ─────────────
        angle_a[0] = float(np.arctan2(T_a_dep[1] - ay, T_a_dep[0] - ax))
        angle_b[0] = float(np.arctan2(T_b_arr[1] - by, T_b_arr[0] - bx))

        # ── Wound section on Roll A ───────────────────────────────────────────
        dtheta = config.REST_LEN / orbit_r_a
        dz     = 2.0 * r * config.REST_LEN / (2.0 * np.pi * orbit_r_a)

        # Compute path lengths to distribute the free-span budget
        T_c_in_3d  = np.array([T_c_in[0],  T_c_in[1],  cz])
        T_c_out_3d = np.array([T_c_out[0], T_c_out[1], cz])
        T_b_arr_3d = np.array([T_b_arr[0], T_b_arr[1], bz])

        span_AC    = float(np.linalg.norm(T_c_in  - T_a_dep))
        arc_len    = orbit_r_c * warp_rad
        span_CB    = float(np.linalg.norm(T_b_arr - T_c_out))
        total_free = max(span_AC + arc_len + span_CB, config.REST_LEN)

        n_free  = max(3, min(int(round(total_free / config.REST_LEN)), N - 3))
        n_wound = N - n_free

        positions: list = []
        for i in range(n_wound):
            theta = angle_a[0] + i * dtheta
            positions.append([ax + orbit_r_a * np.cos(theta),
                               ay + orbit_r_a * np.sin(theta),
                               az + i * dz])

        # Distribute n_free across the 3 free-span segments proportionally
        n_AC  = max(1, min(round(n_free * span_AC  / total_free), n_free - 2))
        n_arc = max(1, min(round(n_free * arc_len  / total_free), n_free - n_AC - 1))
        n_CB  = max(1, n_free - n_AC - n_arc)

        # Segment 1: last wound particle → guide tangent-in
        p_dep = np.array(positions[-1])
        for i in range(1, n_AC + 1):
            t = i / n_AC
            positions.append(list(p_dep + t * (T_c_in_3d - p_dep)))

        # Segment 2: arc on guide orbit surface
        for i in range(1, n_arc + 1):
            t     = i / n_arc
            theta = theta_in + wrap_dir * warp_rad * t
            positions.append([cx + orbit_r_c * np.cos(theta),
                               cy + orbit_r_c * np.sin(theta),
                               cz])

        # Segment 3: guide tangent-out → Roll B
        p_arc_end = np.array(positions[-1])
        for i in range(1, n_CB + 1):
            t = i / n_CB
            positions.append(list(p_arc_end + t * (T_b_arr_3d - p_arc_end)))

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

    mesh_a    = _cyl("roll_a_x",    "roll_a_y",    "roll_a_z",    "roll_a_radius")
    mesh_b    = _cyl("roll_b_x",    "roll_b_y",    "roll_b_z",    "roll_b_radius")
    mesh_mid  = _cyl("cyl_x",       "cyl_y",       "cyl_z",       "cyl_radius")
    mesh_scA  = _cyl("sensor_a_x",  "sensor_a_y",  "sensor_a_z",  "sensor_a_cyl_r")
    mesh_scB  = _cyl("sensor_b_x",  "sensor_b_y",  "sensor_b_z",  "sensor_b_cyl_r")

    obs_a    = ObstacleGPU(mesh_a,   device)
    obs_b    = ObstacleGPU(mesh_b,   device)
    obs_mid  = ObstacleGPU(mesh_mid, device)
    obs_scA  = ObstacleGPU(mesh_scA,  device)
    obs_scB  = ObstacleGPU(mesh_scB,  device)

    # ── Yarn GPU arrays ───────────────────────────────────────────────────────
    pos_np, n_wound = make_initial_positions()

    # Roll rotational state on GPU — updated each substep by their respective kernels.
    angle_a_wp = wp.array([angle_a[0]], dtype=float, device=device)
    omega_a_wp = wp.array([0.0],        dtype=float, device=device)
    angle_b_wp = wp.array([angle_b[0]], dtype=float, device=device)
    # Servo command (size-1 wp.array so updates from on_timer propagate through
    # a captured CUDA graph without rebuild — kernel dereferences omega_cmd[0]
    # at replay time, not capture time).
    omega_cmd_wp = wp.array([0.0], dtype=float, device=device)

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
    vf_a    = VFContacts(N, device);   ee_a    = EEContacts(N - 1, device)
    vf_b    = VFContacts(N, device);   ee_b    = EEContacts(N - 1, device)
    vf_mid  = VFContacts(N, device);   ee_mid  = EEContacts(N - 1, device)
    vf_scA  = VFContacts(N, device);   ee_scA  = EEContacts(N - 1, device)
    vf_scB  = VFContacts(N, device);   ee_scB  = EEContacts(N - 1, device)

    # Self-collision contact array (yarn vs. yarn).
    self_ee = SelfEEContacts(N - 1, device)

    # ── Conservative-bound redetection buffers (Phase 4) ─────────────────────
    pos_det_wp  = wp.zeros(N, dtype=wp.vec3, device=device)
    max_disp_buf = wp.zeros(1, dtype=float,   device=device)
    _force_redetect = [True]

    # contacts: Roll A (index 0, special friction), Roll B + main guide (index 1+, full friction).
    # frictionless_contacts: projection + normal damping only, no friction kernels.
    # Each entry is [obstacle, vf_contacts, ee_contacts, enabled_key].
    contacts = [
        [obs_a,   vf_a,   ee_a],
        [obs_b,   vf_b,   ee_b],
        [obs_mid, vf_mid, ee_mid],
    ]
    frictionless_contacts = [
        [obs_scA,  vf_scA,  ee_scA,  "sensor_a_cyl_enabled"],
        [obs_scB,  vf_scB,  ee_scB,  "sensor_b_cyl_enabled"],
    ]

    # ── Particle-count reinit ─────────────────────────────────────────────────
    # Called when the num_particles slider changes. Rebinds all N-dependent GPU
    # arrays via nonlocal so that every other closure sees the new arrays on its
    # next call — no changes required in _execute_substeps or sim_reset.

    def do_reinit(new_N: int):
        nonlocal N, n_even, n_odd, n_bend, n_wound
        nonlocal pos_wp, vel_wp, prev_pos_wp, inv_mass_wp, yarn_edges_wp
        nonlocal angle_a_wp, omega_a_wp, angle_b_wp, omega_cmd_wp
        nonlocal vf_a, ee_a, vf_b, ee_b, vf_mid, ee_mid, self_ee, contacts
        nonlocal vf_scA, ee_scA, vf_scB, ee_scB, frictionless_contacts
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
        omega_cmd_wp = wp.array([0.0],      dtype=float, device=device)

        pos_np, n_wound = make_initial_positions()
        # make_initial_positions() sets angle_a[0]/angle_b[0] to exact tangent angles — re-sync GPU.
        angle_a_wp  = wp.array([angle_a[0]], dtype=float,   device=device)
        angle_b_wp  = wp.array([angle_b[0]], dtype=float,   device=device)
        _auto_place_sensors()
        pos_wp      = wp.array(pos_np,                               dtype=wp.vec3, device=device)
        vel_wp      = wp.array(np.zeros((N, 3), dtype=np.float32),   dtype=wp.vec3, device=device)
        prev_pos_wp = wp.array(pos_np.copy(),                        dtype=wp.vec3, device=device)
        inv_mass_wp = wp.array(make_inv_mass(),                      dtype=float,   device=device)

        edges_np      = np.stack([np.arange(N - 1), np.arange(1, N)], axis=1).astype(np.int32)
        yarn_edges_wp = wp.array(edges_np, dtype=wp.vec2i, device=device)

        vf_a    = VFContacts(N, device);   ee_a    = EEContacts(N - 1, device)
        vf_b    = VFContacts(N, device);   ee_b    = EEContacts(N - 1, device)
        vf_mid  = VFContacts(N, device);   ee_mid  = EEContacts(N - 1, device)
        vf_scA  = VFContacts(N, device);   ee_scA  = EEContacts(N - 1, device)
        vf_scB  = VFContacts(N, device);   ee_scB  = EEContacts(N - 1, device)
        self_ee = SelfEEContacts(N - 1, device)

        contacts = [
            [contacts[0][0], vf_a,   ee_a],
            [contacts[1][0], vf_b,   ee_b],
            [contacts[2][0], vf_mid, ee_mid],
        ]
        frictionless_contacts = [
            [frictionless_contacts[0][0], vf_scA,  ee_scA,  "sensor_a_cyl_enabled"],
            [frictionless_contacts[1][0], vf_scB,  ee_scB,  "sensor_b_cyl_enabled"],
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
        omega_cmd_wp.assign(wp.array([0.0],      dtype=float, device=device))
        _servo_integral[0] = 0.0
        _servo_omega_cmd[0] = 0.0
        pos0, n_wound = make_initial_positions()
        # make_initial_positions() sets angle_a[0]/angle_b[0] to exact tangent angles — re-sync GPU.
        angle_a_wp.assign(wp.array([angle_a[0]], dtype=float, device=device))
        angle_b_wp.assign(wp.array([angle_b[0]], dtype=float, device=device))
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
            "roll_a_servo_on":        int(state.get("roll_a_servo_on", 0)),
        }

    # ── Tension sensor state (independent sphere windows) ────────────────────
    # Each sensor is a sphere with independently adjustable centre and radius.
    # All particles inside the sphere contribute to the averaged tension reading.
    _sphere_centers = [np.zeros(3), np.zeros(3)]  # [upstream, downstream]
    # Per-sensor 3×3 orientation; columns are local (X=tangent, Y, Z) axes in world.
    _sensor_R = [np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32)]

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

    def _wrap_angle_contacts(pp, dbg=None):
        """Wrap angle θ (rad) by fitting the yarn arc around the guide cylinder.

        Geometric idea: only the part of the yarn that actually wraps the guide
        is at the contact distance r_c = cyl_radius + ogc_r from the axis.  The
        tangent straight segments leading INTO and OUT of the wrap are *farther*
        from the axis and curve outward.  We therefore separate two roles:

          • `cyl_detect_r` defines a WIDE search region around the guide — just
            to locate yarn particles near it (with hysteresis-friendly margin).
          • A SECOND, narrow band at distance ≈ r_c selects which of those
            particles actually lie on the wrap arc.

        Algorithm:
          1. Search: collect all particles within `cyl_detect_r` of the axis
             (XY plane) and within the cylinder's Z range.
          2. Run isolation: among them, take the longest contiguous yarn-index
             run — that's the segment that traverses the guide.
          3. Arc filter: from that run keep only particles whose XY distance to
             axis lies in [r_c − tol, r_c + tol].  Straight tangents are
             excluded because their distance grows linearly away from r_c.
          4. Integrate: |Σ Δθ| (with ±π unwrap) over the arc particles only.

        The result is invariant to `cyl_detect_r` as long as it exceeds
        r_c + tol — only the arc particles contribute, regardless of how big
        the search region is.

        Returns (theta_rad, n_arc).
        """
        cx = float(state["cyl_x"])
        cy = float(state["cyl_y"])
        cz = float(state["cyl_z"])
        r_cyl = float(state["cyl_radius"])
        ogc_r = float(state.get("ogc_r", 0.005))
        rd    = float(state.get("cyl_detect_r", 0.20))

        dx = pp[:, 0] - cx
        dy = pp[:, 1] - cy
        dist_xy = np.sqrt(dx * dx + dy * dy)
        in_z = np.abs(pp[:, 2] - cz) < CYL_HALF_H

        # Step 1: search volume
        in_search = (dist_xy < rd) & in_z
        idxs = np.where(in_search)[0]

        r_contact = r_cyl + ogc_r
        arc_tol = max(3.0 * ogc_r, 0.05 * r_cyl)

        if dbg is not None:
            # Cap for the shared-memory transport
            sel = idxs[:MAX_DEBUG_PARTS]
            dbg["cx"]        = float(cx)
            dbg["cy"]        = float(cy)
            dbg["r_cyl"]     = float(r_cyl)
            dbg["r_contact"] = float(r_contact)
            dbg["arc_tol"]   = float(arc_tol)
            dbg["rd"]        = float(rd)
            dbg["idxs"]      = np.asarray(sel, dtype=np.int32)
            dbg["xs"]        = dx[sel].astype(np.float64)
            dbg["ys"]        = dy[sel].astype(np.float64)
            dbg["rs"]        = dist_xy[sel].astype(np.float64)
            dbg["cls"]       = np.zeros(len(sel), dtype=np.int32)
            dbg["longest_len"] = 0
            dbg["n_arc"]     = 0
            dbg["refined_in_xy"]  = None
            dbg["refined_out_xy"] = None

        if len(idxs) < 2:
            return 0.0, int(len(idxs))

        # Step 2: longest contiguous run of yarn indices
        runs = []
        cur = [int(idxs[0])]
        for i in idxs[1:]:
            i = int(i)
            if i == cur[-1] + 1:
                cur.append(i)
            else:
                runs.append(cur)
                cur = [i]
        runs.append(cur)
        longest = np.asarray(max(runs, key=len), dtype=int)

        if dbg is not None:
            dbg["longest_len"] = int(len(longest))
            # Upgrade cls=1 for any debug particle that's part of the longest run
            longest_set = set(int(i) for i in longest)
            sel_idxs = dbg["idxs"]
            for k, i in enumerate(sel_idxs):
                if int(i) in longest_set:
                    dbg["cls"][k] = 1

        if len(longest) < 2:
            return 0.0, int(len(longest))

        # Step 3: arc filter — distance ≈ contact radius
        # Tolerance: thick enough to absorb numerical jitter at the OGC
        # equilibrium distance, thin enough to exclude tangent segments that
        # curve away from the cylinder.
        arc_mask = (dist_xy[longest] >= r_contact - arc_tol) & \
                   (dist_xy[longest] <= r_contact + arc_tol)
        arc_idxs = longest[arc_mask]

        if dbg is not None:
            dbg["n_arc"] = int(len(arc_idxs))
            arc_set = set(int(i) for i in arc_idxs)
            sel_idxs = dbg["idxs"]
            for k, i in enumerate(sel_idxs):
                if int(i) in arc_set:
                    dbg["cls"][k] = 2

        if len(arc_idxs) < 2:
            return 0.0, int(len(arc_idxs))

        # Step 4: angles of arc particles around the cylinder axis (XY plane).
        angles = np.arctan2(dy[arc_idxs], dx[arc_idxs]).astype(float).copy()

        # Step 5: refine the boundary angles by intersecting the segment
        # (arc-boundary particle ↔ its out-of-band yarn neighbor) with the
        # contact circle r = r_contact.  That intersection is the true tangent
        # point regardless of whether the boundary particle is on the tangent
        # line (R > r_c) or already on the arc with small jitter (R ≈ r_c) —
        # we don't have to guess.  Solving
        #   |(1−t)·p_neighbor + t·p_boundary|² = r_contact²
        # gives a quadratic in t; we pick the root nearest to the arc end of
        # the segment, which is the physically-correct crossing.
        def _refine(i_bdy, i_nbr):
            if i_nbr < 0 or i_nbr >= len(pp):
                return None
            xn = float(pp[i_nbr, 0]) - cx
            yn = float(pp[i_nbr, 1]) - cy
            xb = float(dx[i_bdy])
            yb = float(dy[i_bdy])
            vx = xb - xn;  vy = yb - yn
            A = vx * vx + vy * vy
            if A < 1e-12:
                return None
            B = 2.0 * (xn * vx + yn * vy)
            C = xn * xn + yn * yn - r_contact * r_contact
            disc = B * B - 4.0 * A * C
            if disc < 0.0:
                return None       # line doesn't cross the contact circle
            sq = float(np.sqrt(disc))
            t1 = (-B - sq) / (2.0 * A)
            t2 = (-B + sq) / (2.0 * A)
            # Prefer the root in (0, 1] (between neighbor and boundary); if
            # neither fits, allow modest extrapolation toward the boundary
            # (t slightly > 1) to handle the case where both particles are
            # just above the contact circle.
            candidates = sorted([t1, t2], key=lambda v: abs(v - 1.0))
            for t in candidates:
                if -0.1 <= t <= 1.3:
                    xt = xn + t * vx
                    yt = yn + t * vy
                    return float(np.arctan2(yt, xt))
            return None

        new_in  = _refine(int(arc_idxs[0]),  int(arc_idxs[0])  - 1)
        new_out = _refine(int(arc_idxs[-1]), int(arc_idxs[-1]) + 1)
        if new_in  is not None:
            angles[0]  = new_in
        if new_out is not None:
            angles[-1] = new_out

        if dbg is not None:
            if new_in is not None:
                dbg["refined_in_xy"]  = (float(r_contact * np.cos(new_in)),
                                         float(r_contact * np.sin(new_in)))
            if new_out is not None:
                dbg["refined_out_xy"] = (float(r_contact * np.cos(new_out)),
                                         float(r_contact * np.sin(new_out)))

        # Step 6: signed Δθ sum with ±π unwrap → robust net rotation
        dth = np.diff(angles)
        dth = (dth + np.pi) % (2.0 * np.pi) - np.pi
        signed = float(np.sum(dth))
        wrap = float(abs(signed))
        if dbg is not None:
            dbg["signed_theta"] = signed
            # Keep the post-refinement arc-particle azimuths so the viz can
            # draw a radial tick at each one — that's the discrete sampling
            # the algorithm sums Δθ across.
            dbg["arc_angles"] = angles.copy()
        return wrap, int(len(arc_idxs))

    def _write_shared(pp, sim_t):
        """Compute tension via detection spheres, then Capstan metrics → shared[]."""
        sc_a = np.array([float(state["sensor_a_x"]),
                         float(state["sensor_a_y"]),
                         float(state["sensor_a_z"])])
        sc_b = np.array([float(state["sensor_b_x"]),
                         float(state["sensor_b_y"]),
                         float(state["sensor_b_z"])])
        _sphere_centers[0] = sc_a
        _sphere_centers[1] = sc_b

        # Oriented box: project (p - center) onto each sensor's local axes
        # (columns of _sensor_R) before the half-extent test.
        ha = (float(state["sensor_a_hx"]), float(state["sensor_a_hy"]), float(state["sensor_a_hz"]))
        hb = (float(state["sensor_b_hx"]), float(state["sensor_b_hy"]), float(state["sensor_b_hz"]))
        la = np.abs((pp - sc_a) @ _sensor_R[0])
        lb = np.abs((pp - sc_b) @ _sensor_R[1])
        mask_a = (la[:, 0] < ha[0]) & (la[:, 1] < ha[1]) & (la[:, 2] < ha[2])
        mask_b = (lb[:, 0] < hb[0]) & (lb[:, 1] < hb[1]) & (lb[:, 2] < hb[2])

        T_a   = _tension_from_mask(pp, mask_a)
        T_b   = _tension_from_mask(pp, mask_b)
        dbg   = {}
        theta, n_contact = _wrap_angle_contacts(pp, dbg=dbg)
        mu_k  = float(state["mu_kinetic"])
        capstan_pred = T_a * np.exp(mu_k * theta)
        residual     = (T_b / capstan_pred) if capstan_pred > 1e-9 else 0.0
        # shared layout: [T_a, T_b, theta_deg, capstan_pred, residual, sim_time, n_contact]
        shared[0] = T_a
        shared[1] = T_b
        shared[2] = float(np.degrees(theta))
        shared[3] = capstan_pred
        shared[4] = residual
        shared[5] = sim_t
        shared[6] = float(n_contact)

        # Wrap-angle debug snapshot for the live 2D viz.
        n_dbg = int(len(dbg.get("xs", [])))
        dbg_shared[0]  = float(n_dbg)
        dbg_shared[1]  = float(dbg.get("cx", 0.0))
        dbg_shared[2]  = float(dbg.get("cy", 0.0))
        dbg_shared[3]  = float(dbg.get("r_cyl", 0.0))
        dbg_shared[4]  = float(dbg.get("r_contact", 0.0))
        dbg_shared[5]  = float(dbg.get("arc_tol", 0.0))
        dbg_shared[6]  = float(dbg.get("rd", 0.0))
        dbg_shared[7]  = float(theta)
        dbg_shared[8]  = float(n_contact)
        dbg_shared[9]  = float(dbg.get("longest_len", 0))
        rin  = dbg.get("refined_in_xy")
        rout = dbg.get("refined_out_xy")
        nan  = float("nan")
        dbg_shared[10] = rin[0]  if rin  else nan
        dbg_shared[11] = rin[1]  if rin  else nan
        dbg_shared[12] = rout[0] if rout else nan
        dbg_shared[13] = rout[1] if rout else nan
        dbg_shared[14] = float(sim_t)
        dbg_shared[15] = float(dbg.get("signed_theta", 0.0))
        if n_dbg > 0:
            xs  = dbg["xs"];  ys = dbg["ys"];  rs = dbg["rs"];  cls = dbg["cls"]
            base = DBG_HEADER_LEN
            for k in range(n_dbg):
                off = base + DBG_STRIDE * k
                dbg_shared[off + 0] = float(xs[k])
                dbg_shared[off + 1] = float(ys[k])
                dbg_shared[off + 2] = float(rs[k])
                dbg_shared[off + 3] = float(cls[k])

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
        servo_on   = bool(int(state.get("roll_a_servo_on", 0)))

        for _ in range(config.SUBSTEPS):
            if servo_on:
                roll_a_servo_step(
                    pos_wp, center_a, orbit_r_a, sub_dt, 200.0,
                    angle_a_wp, omega_a_wp, omega_cmd_wp, device,
                )
            else:
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
            for obs, vf, ee, ekey in frictionless_contacts:
                if bool(int(state.get(ekey, 0))):
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
                for obs, vf, ee, ekey in frictionless_contacts:
                    if bool(int(state.get(ekey, 0))):
                        project_vf(pos_wp, inv_mass_wp, vf, r, stiff, device)
                        project_ee(pos_wp, inv_mass_wp, yarn_edges_wp, ee, r, stiff, device)
                if self_coll_on:
                    project_self_ee(pos_wp, inv_mass_wp, yarn_edges_wp,
                                    self_ee, r, self_ee_stiff, device)
            # Roll A (contacts[0]): friction only on free-span particles.
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
            for obs, vf, ee, ekey in frictionless_contacts:
                if bool(int(state.get(ekey, 0))):
                    damp_normal_velocity(vel_wp, inv_mass_wp, vf, r, device)
            clamp_velocity(vel_wp, v_max, device)

    def _detect_contacts():
        """Seed contact arrays once (on init/reset) before the substep loop runs."""
        r = float(state["ogc_r"])
        self_coll = bool(int(state.get("self_collision", 1)))
        for obs, vf, ee in contacts:
            detect_vertex_facet(pos_wp, obs, vf, r, device)
            detect_edge_edge(pos_wp, yarn_edges_wp, obs, ee, r, device)
        for obs, vf, ee, ekey in frictionless_contacts:
            if bool(int(state.get(ekey, 0))):
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

    va,   fa   = mesh_for_render(mesh_a)
    vb,   fb   = mesh_for_render(mesh_b)
    vm,   fm   = mesh_for_render(mesh_mid)
    vsA,  fsA  = mesh_for_render(mesh_scA)
    vsB,  fsB  = mesh_for_render(mesh_scB)

    # Sensor cylinder colour: solid, matches sensor dot colour
    SCY_A_COL = (1.0, 0.85, 0.0, 1.0)   # yellow — sensor A
    SCY_B_COL = (0.0, 0.85, 0.25, 1.0)  # green  — sensor B

    roll_a_vis  = visuals.Mesh(vertices=va,  faces=fa,  color=ROLL_A_COL,  shading="smooth", parent=view.scene)
    roll_b_vis  = visuals.Mesh(vertices=vb,  faces=fb,  color=ROLL_B_COL,  shading="smooth", parent=view.scene)
    cyl_vis     = visuals.Mesh(vertices=vm,  faces=fm,  color=CYL_COL,     shading="smooth", parent=view.scene)
    scA_vis     = visuals.Mesh(vertices=vsA, faces=fsA, color=SCY_A_COL,   shading="smooth", parent=view.scene)
    scB_vis     = visuals.Mesh(vertices=vsB, faces=fsB, color=SCY_B_COL,   shading="smooth", parent=view.scene)
    scA_vis.visible  = bool(int(state.get("sensor_a_cyl_enabled", 0)))
    scB_vis.visible  = bool(int(state.get("sensor_b_cyl_enabled", 0)))

    # Wrap-angle detection volume: a translucent cylinder of radius cyl_detect_r
    # around the guide.  Yarn particles inside this volume are summed into θ.
    # Hidden by default — toggle via the "Show wrap-detect volume" checkbox.
    mesh_detect = _cyl("cyl_x", "cyl_y", "cyl_z", "cyl_detect_r")
    vd, fd = mesh_for_render(mesh_detect)
    cyl_detect_vis = visuals.Mesh(vertices=vd, faces=fd, color=CYL_DET_COL,
                                  shading="smooth", parent=view.scene)
    cyl_detect_vis.set_gl_state("translucent", depth_test=True)
    cyl_detect_vis.visible = bool(int(state.get("cyl_detect_show", 0)))

    p = pos_wp.numpy()
    yarn_colors = make_yarn_colors(N)
    yarn_line   = visuals.Line(pos=p, color=yarn_colors, width=3,
                               connect="strip", parent=view.scene)
    marker_a = visuals.Markers(parent=view.scene)
    marker_b = visuals.Markers(parent=view.scene)
    marker_a.set_data(p[:1],  face_color=ANCHOR_COL, size=14, edge_width=0)
    marker_b.set_data(p[-1:], face_color=PULL_COL,   size=14, edge_width=0)

    # Tension sensor boxes: axis-aligned plates whose half-extents in X, Y, Z
    # are user-tunable.  The detection mask is a point-in-AABB test using the
    # same half-extents, so the rendered volume matches what's being measured.
    _sc_a0 = np.array([DEFAULTS["sensor_a_x"], DEFAULTS["sensor_a_y"], DEFAULTS["sensor_a_z"]], dtype=np.float32)
    _sc_b0 = np.array([DEFAULTS["sensor_b_x"], DEFAULTS["sensor_b_y"], DEFAULTS["sensor_b_z"]], dtype=np.float32)
    _sphere_centers[0] = _sc_a0;  _sphere_centers[1] = _sc_b0

    # Cube face triangulation (shared between both sensors).
    _BOX_FACES = np.array([
        # -Y, +Y
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        # -Z, +Z
        [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2],
        # -X, +X
        [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
    ], dtype=np.uint32)

    def _box_verts(center, hx, hy, hz, R=None):
        c = np.asarray(center, dtype=np.float32).reshape(3)
        local = np.array([
            [-hx, -hy, -hz], [ hx, -hy, -hz],
            [ hx, -hy,  hz], [-hx, -hy,  hz],
            [-hx,  hy, -hz], [ hx,  hy, -hz],
            [ hx,  hy,  hz], [-hx,  hy,  hz],
        ], dtype=np.float32)
        if R is not None:
            local = local @ np.asarray(R, dtype=np.float32).T
        return (local + c[None, :]).astype(np.float32)

    _SENSOR_A_RGB = (1.0, 0.85, 0.0)
    _SENSOR_B_RGB = (0.0, 0.85, 0.25)

    sensor_box_a = visuals.Mesh(
        vertices=_box_verts(_sc_a0, DEFAULTS["sensor_a_hx"], DEFAULTS["sensor_a_hy"], DEFAULTS["sensor_a_hz"]),
        faces=_BOX_FACES,
        color=_SENSOR_A_RGB + (DEFAULTS["sensor_a_alpha"],),
        shading="smooth", parent=view.scene)
    sensor_box_b = visuals.Mesh(
        vertices=_box_verts(_sc_b0, DEFAULTS["sensor_b_hx"], DEFAULTS["sensor_b_hy"], DEFAULTS["sensor_b_hz"]),
        faces=_BOX_FACES,
        color=_SENSOR_B_RGB + (DEFAULTS["sensor_b_alpha"],),
        shading="smooth", parent=view.scene)
    sensor_box_a.set_gl_state("translucent", depth_test=True)
    sensor_box_b.set_gl_state("translucent", depth_test=True)

    # Small opaque centre dots — visible through the transparent mesh.
    sensor_dot_a = visuals.Markers(parent=view.scene)
    sensor_dot_b = visuals.Markers(parent=view.scene)
    sensor_dot_a.set_data(_sc_a0[np.newaxis], face_color=(1.0, 0.9, 0.0, 1.0), size=7, edge_width=0)
    sensor_dot_b.set_data(_sc_b0[np.newaxis], face_color=(0.0, 0.9, 0.3, 1.0), size=7, edge_width=0)

    def _update_sensor_visuals():
        sa = _sphere_centers[0];  sb = _sphere_centers[1]
        ha = (float(state.get("sensor_a_hx", 0.05)),
              float(state.get("sensor_a_hy", 0.005)),
              float(state.get("sensor_a_hz", 0.05)))
        hb = (float(state.get("sensor_b_hx", 0.05)),
              float(state.get("sensor_b_hy", 0.005)),
              float(state.get("sensor_b_hz", 0.05)))
        aa = float(np.clip(state.get("sensor_a_alpha", 0.4), 0.0, 1.0))
        ab = float(np.clip(state.get("sensor_b_alpha", 0.4), 0.0, 1.0))
        sensor_box_a.set_data(vertices=_box_verts(sa, *ha, R=_sensor_R[0]), faces=_BOX_FACES,
                              color=_SENSOR_A_RGB + (aa,))
        sensor_box_b.set_data(vertices=_box_verts(sb, *hb, R=_sensor_R[1]), faces=_BOX_FACES,
                              color=_SENSOR_B_RGB + (ab,))
        sensor_dot_a.set_data(np.array([sa], dtype=np.float32),
                              face_color=(1.0, 0.9, 0.0, 1.0), size=7, edge_width=0)
        sensor_dot_b.set_data(np.array([sb], dtype=np.float32),
                              face_color=(0.0, 0.9, 0.3, 1.0), size=7, edge_width=0)

    _update_sensor_visuals()
    _auto_place_sensors()   # set sensor positions from geometry at scene startup

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
        _graph[0] = None
        sim_reset()
        _auto_place_sensors()

    def rebuild_roll_b():
        new_mesh = _cyl("roll_b_x", "roll_b_y", "roll_b_z", "roll_b_radius")
        contacts[1][0] = ObstacleGPU(new_mesh, device)
        vv, ff = mesh_for_render(new_mesh)
        roll_b_vis.set_data(vertices=vv, faces=ff, color=ROLL_B_COL)
        _graph[0] = None
        sim_reset()
        _auto_place_sensors()

    def rebuild_guide():
        new_mesh = _cyl("cyl_x", "cyl_y", "cyl_z", "cyl_radius")
        contacts[2][0] = ObstacleGPU(new_mesh, device)
        vv, ff = mesh_for_render(new_mesh)
        cyl_vis.set_data(vertices=vv, faces=ff, color=CYL_COL)
        _graph[0] = None
        sim_reset()
        _auto_place_sensors()

    def rebuild_detect_vol():
        """Refresh the wrap-angle detection cylinder (visual only; no physics)."""
        new_mesh = _cyl("cyl_x", "cyl_y", "cyl_z", "cyl_detect_r")
        vv, ff = mesh_for_render(new_mesh)
        cyl_detect_vis.set_data(vertices=vv, faces=ff, color=CYL_DET_COL)

    def rebuild_scA():
        new_mesh = _cyl("sensor_a_x", "sensor_a_y", "sensor_a_z", "sensor_a_cyl_r")
        frictionless_contacts[0][0] = ObstacleGPU(new_mesh, device)
        vv, ff = mesh_for_render(new_mesh)
        scA_vis.set_data(vertices=vv, faces=ff, color=SCY_A_COL)
        _graph[0] = None

    def rebuild_scB():
        new_mesh = _cyl("sensor_b_x", "sensor_b_y", "sensor_b_z", "sensor_b_cyl_r")
        frictionless_contacts[1][0] = ObstacleGPU(new_mesh, device)
        vv, ff = mesh_for_render(new_mesh)
        scB_vis.set_data(vertices=vv, faces=ff, color=SCY_B_COL)
        _graph[0] = None

    # ── Sensor gizmo — keyboard select, mouse drag to move ───────────────────────
    # vispy is not a 3D editor; reliable click-picking of small 3D objects is not
    # supported. Instead: press A or B to select a sensor, drag to move it, hold
    # X / Y / Z while dragging to constrain to that world axis, Escape to deselect.
    #
    # Gizmo arrows (X=red, Y=green, Z=blue) are shown as visual reference only.
    # Camera orbit still works whenever no drag is in progress.

    _GIZMO_LEN = 0.30

    _AX_DIRS = {
        "X": np.array([1.0, 0.0, 0.0]),
        "Y": np.array([0.0, 1.0, 0.0]),
        "Z": np.array([0.0, 0.0, 1.0]),
    }
    _AX_COLS = {
        "X": np.array([0.95, 0.20, 0.20, 1.0]),
        "Y": np.array([0.20, 0.90, 0.20, 1.0]),
        "Z": np.array([0.25, 0.50, 1.00, 1.0]),
    }
    _AX_HOT = np.array([1.0, 0.85, 0.0, 1.0])

    _sel = {
        "who":       None,   # "A" or "B" — selected sensor
        "axis_lock": None,   # "X","Y","Z" or None — held key constraint
        "dragging":  False,
        "drag_start_mouse": None,
        "drag_start_world": None,
    }

    def _sensor_center(who):
        return np.array(_sphere_centers[0 if who == "sensor_a" else 1], dtype=float)

    def _object_center(who):
        """Return world-space centre for any pickable object."""
        if who == "sensor_a":
            return np.array(_sphere_centers[0], dtype=float)
        if who == "sensor_b":
            return np.array(_sphere_centers[1], dtype=float)
        key_map = {
            "roll_a": ("roll_a_x", "roll_a_y", "roll_a_z"),
            "roll_b": ("roll_b_x", "roll_b_y", "roll_b_z"),
            "cyl":    ("cyl_x",    "cyl_y",    "cyl_z"),
        }
        kx, ky, kz = key_map[who]
        return np.array([state[kx], state[ky], state[kz]], dtype=float)

    # Single shared gizmo — repositioned to whichever object is selected.
    gizmo_line = visuals.Line(pos=np.zeros((6, 3), dtype=np.float32),
                              color=np.ones((6, 4), dtype=np.float32),
                              width=3, connect="segments", parent=view.scene)
    gizmo_line.visible = False

    def _refresh_gizmos():
        if _sel["who"] is None:
            gizmo_line.visible = False
            return
        c    = _object_center(_sel["who"])
        lock = _sel["axis_lock"]
        pts  = np.empty((6, 3), dtype=np.float32)
        cols = np.empty((6, 4), dtype=np.float32)
        for i, name in enumerate(["X", "Y", "Z"]):
            tip  = (c + _AX_DIRS[name] * _GIZMO_LEN).astype(np.float32)
            col  = _AX_HOT if name == lock else _AX_COLS[name]
            pts[i*2] = c;    pts[i*2+1] = tip
            cols[i*2] = col; cols[i*2+1] = col
        gizmo_line.set_data(pos=pts, color=cols)
        gizmo_line.visible = True

    def _screen_delta_to_world(d_screen):
        """Convert a 2-element screen-pixel delta to a world-space 3-D vector.

        Uses the exact same formula that vispy TurntableCamera uses for
        Shift+drag panning, so the result is guaranteed to match the rendered
        view regardless of camera orientation.

        d_screen: (dx, dy) in screen pixels, positive = right / down.
        Returns a world-space np.ndarray([dx, dy, dz]).
        """
        cam   = view.camera
        norm  = float(np.mean(cam._viewbox.size))
        sf    = float(cam._scale_factor)
        dist  = np.array([-d_screen[0], d_screen[1]], dtype=float) / norm * sf
        dx_l, dy_l, dz_l = cam._dist_to_trans(dist)
        ff               = cam._flip_factors
        up, forward, right = cam._get_dim_vectors()
        world = right * dx_l + forward * dy_l + up * dz_l
        world = np.array([ff[0] * world[0], ff[1] * world[1], ff[2] * world[2]])
        return -world

    def _cam_axes():
        """Return (right, up, forward) unit vectors in world space.

        Derived from _screen_delta_to_world so they are always consistent
        with the pan formula — no trigonometry errors possible.
          right_w   = direction objects move when cursor goes right (+screen X)
          up_w      = direction objects move when cursor goes up   (-screen Y)
          forward_w = cross(right_w, up_w), pointing into the scene
        """
        r  = _screen_delta_to_world(np.array([1.0, 0.0]))
        dn = _screen_delta_to_world(np.array([0.0, 1.0]))  # screen down → world down
        r  /= np.linalg.norm(r)  + 1e-12
        u   = -dn / (np.linalg.norm(dn) + 1e-12)          # flip to world up
        f   = np.cross(r, u)
        f  /= np.linalg.norm(f)  + 1e-12
        return r, u, f

    # ── Ray–geometry primitives ───────────────────────────────────────────────

    def _ray_sphere(ro, rd, center, radius):
        """Return smallest positive t for ray-sphere intersection, or None."""
        oc = ro - np.array(center, dtype=float)
        b  = 2.0 * np.dot(oc, rd)
        c  = np.dot(oc, oc) - radius * radius
        d  = b * b - 4.0 * c
        if d < 0:
            return None
        sq = np.sqrt(d)
        t  = (-b - sq) / 2.0
        if t < 1e-4:
            t = (-b + sq) / 2.0
        return t if t > 1e-4 else None

    def _ray_aabb(ro, rd, center, hx, hy, hz):
        """Return smallest positive t for ray vs axis-aligned box, or None.

        Standard slab method.  center is (cx, cy, cz); the box spans
        [cx-hx, cx+hx] × [cy-hy, cy+hy] × [cz-hz, cz+hz].
        """
        mins = (center[0] - hx, center[1] - hy, center[2] - hz)
        maxs = (center[0] + hx, center[1] + hy, center[2] + hz)
        t_lo, t_hi = -1.0e30, 1.0e30
        for i in range(3):
            if abs(rd[i]) < 1.0e-12:
                if ro[i] < mins[i] or ro[i] > maxs[i]:
                    return None
                continue
            inv = 1.0 / rd[i]
            t1 = (mins[i] - ro[i]) * inv
            t2 = (maxs[i] - ro[i]) * inv
            if t1 > t2:
                t1, t2 = t2, t1
            if t1 > t_lo:
                t_lo = t1
            if t2 < t_hi:
                t_hi = t2
        if t_hi < t_lo or t_hi < 1.0e-4:
            return None
        return t_lo if t_lo > 1.0e-4 else t_hi

    def _ray_zcylinder(ro, rd, center, radius, half_h):
        """Return smallest positive t for ray against a Z-aligned capped cylinder,
        or None.  center is (cx,cy,cz); axis runs from cz-half_h to cz+half_h."""
        cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
        # Infinite cylinder (ignore Z) – 2-D problem in XY
        ox, oy = ro[0] - cx, ro[1] - cy
        dx, dy = rd[0], rd[1]
        a  = dx*dx + dy*dy
        t_best = None
        if a > 1e-12:
            b  = 2.0 * (ox*dx + oy*dy)
            c  = ox*ox + oy*oy - radius*radius
            d  = b*b - 4.0*a*c
            if d >= 0:
                sq = np.sqrt(d)
                for t in ((-b - sq) / (2*a), (-b + sq) / (2*a)):
                    if t > 1e-4:
                        z_hit = ro[2] + t * rd[2]
                        if cz - half_h <= z_hit <= cz + half_h:
                            if t_best is None or t < t_best:
                                t_best = t
        # End caps (z = cz ± half_h planes)
        if abs(rd[2]) > 1e-12:
            for cap_z in (cz - half_h, cz + half_h):
                t = (cap_z - ro[2]) / rd[2]
                if t > 1e-4:
                    x_hit = ro[0] + t * rd[0] - cx
                    y_hit = ro[1] + t * rd[1] - cy
                    if x_hit*x_hit + y_hit*y_hit <= radius*radius:
                        if t_best is None or t < t_best:
                            t_best = t
        return t_best

    def _camera_ray(sx, sy):
        """Return (cam_pos, ray_dir) for screen pixel (sx, sy)."""
        right_w, up_w, forward_w = _cam_axes()
        ctr     = np.array(view.camera.center, dtype=float)
        cam_pos = ctr - float(view.camera.distance) * forward_w
        cw, ch  = canvas.size
        half_h  = np.tan(np.radians(view.camera.fov) / 2.0)
        ndx = (2.0 * sx / cw - 1.0) * (cw / ch) * half_h
        ndy = (1.0 - 2.0 * sy / ch) * half_h
        ray_dir = forward_w + right_w * ndx + up_w * ndy
        ray_dir /= np.linalg.norm(ray_dir) + 1e-12
        return cam_pos, ray_dir

    # Minimum clickable radius in screen pixels (object selection threshold).
    _MIN_HIT_PX = 14

    # Camera-pos verification flag — print a one-time comparison on first pick
    _verified_cam = [False]

    def _camera_ray_vispy(sx, sy):
        """Construct the world-space ray for canvas pixel (sx, sy) using vispy's
        actual camera transform.

        Why this is correct: vispy's `cam.transform` is the affine that maps
        camera-local coordinates to world coordinates (camera-to-world).  We
        build the ray in camera-local space (origin at 0, target at one unit
        in front through the pixel) and apply the transform to both points.
        The resulting (ro, rd) is exactly the ray vispy uses to render.

        This avoids re-deriving cam_pos / forward / right / up analytically,
        which is brittle: vispy's TurntableCamera composes T(center)·R(az)·
        R(90+elevation,-X)·R(roll)·T(0,0,-scale_factor), and any sign or
        axis-flip mismatch in the analytical derivation produces a ray that
        doesn't match the rendering — which is exactly what was happening.
        """
        cam = view.camera
        cw, ch = canvas.size
        fov_rad = np.radians(cam.fov)
        aspect  = cw / ch
        half_tan = np.tan(fov_rad / 2.0)

        # Pixel → NDC, OpenGL convention (+Y up, -Z forward)
        ndc_x = 2.0 * sx / cw - 1.0
        ndc_y = 1.0 - 2.0 * sy / ch

        # In camera-local space: origin and a point at depth 1 in front of camera
        p0_local = np.array([0.0, 0.0, 0.0])
        p1_local = np.array([ndc_x * aspect * half_tan, ndc_y * half_tan, -1.0])

        # Apply vispy's camera-to-world transform
        p0_world = np.asarray(cam.transform.map(p0_local))[:3]
        p1_world = np.asarray(cam.transform.map(p1_local))[:3]

        ro = p0_world.astype(float)
        rd = (p1_world - p0_world).astype(float)
        rd /= np.linalg.norm(rd) + 1e-12
        return ro, rd

    def _pick_world_pos(sx, sy, plane_pt=None):
        """Cast a true world-space ray through pixel (sx, sy) and return the
        front-most object intersection.

        Picking pipeline:
          1. Build ro, rd using vispy's actual camera transform.
          2. For each object, intersect ray with its true geometry expanded
             by pick_r (= max(physical, _MIN_HIT_PX pixels in world units)).
          3. Sort hits by t (front-most wins) and return that object plus
             the surface hit point of its true (un-expanded) geometry.

        No screen-space heuristics, no axis projection, no height caps — the
        ray either intersects a 3D object or it doesn't.
        """
        ro, rd = _camera_ray_vispy(sx, sy)
        cam = view.camera
        cw, ch  = canvas.size
        half_tan = np.tan(np.radians(cam.fov) / 2.0)

        # One-time sanity check: compare our previously-analytical cam_pos with
        # the one vispy actually uses.  If they differ, the analytical version
        # was wrong (which is why picking was broken).
        if not _verified_cam[0]:
            try:
                right_w, up_w, forward_w = _cam_axes()
                ctr_a = np.array(cam.center, dtype=float)
                cam_a = ctr_a - float(cam.distance) * forward_w
                print(f"[pick] cam_pos analytical={cam_a}  vispy={ro}", flush=True)
            except Exception as e:
                print(f"[pick] cam_pos check skipped: {e}", flush=True)
            _verified_cam[0] = True

        def _pick_r(center, world_r):
            """World-space pick radius = max(physical, _MIN_HIT_PX pixels at object distance)."""
            dist = float(np.linalg.norm(np.array(center, float) - ro))
            one_px = max(dist * 2.0 * half_tan / ch, 1e-6)
            return max(_MIN_HIT_PX * one_px, world_r)

        # All pickable objects: (center, geom_data, name, shape)
        #   shape == "box"   → geom_data = (hx, hy, hz)
        #   shape == "cyl"   → geom_data = world_r (scalar)
        sa_hs = (float(state.get("sensor_a_hx", 0.05)),
                 float(state.get("sensor_a_hy", 0.005)),
                 float(state.get("sensor_a_hz", 0.05)))
        sb_hs = (float(state.get("sensor_b_hx", 0.05)),
                 float(state.get("sensor_b_hy", 0.005)),
                 float(state.get("sensor_b_hz", 0.05)))
        objects = [
            (tuple(_sphere_centers[0]), sa_hs, "sensor_a", "box"),
            (tuple(_sphere_centers[1]), sb_hs, "sensor_b", "box"),
            ((state["roll_a_x"], state["roll_a_y"], state["roll_a_z"]),
                float(state["roll_a_radius"]), "roll_a", "cyl"),
            ((state["roll_b_x"], state["roll_b_y"], state["roll_b_z"]),
                float(state["roll_b_radius"]), "roll_b", "cyl"),
            ((state["cyl_x"], state["cyl_y"], state["cyl_z"]),
                float(state["cyl_radius"]), "cyl", "cyl"),
        ]

        candidates = []  # (t, name, center, geom_data, shape)
        for center, geom_data, name, shape in objects:
            if shape == "cyl":
                pr = _pick_r(center, geom_data)
                t = _ray_zcylinder(ro, rd, center, pr, float(CYL_HALF_H))
            else:  # box
                hx, hy, hz = geom_data
                # Pixel-size margin: expand each half-extent so tiny boxes stay clickable
                dist = float(np.linalg.norm(np.array(center, float) - ro))
                one_px = max(dist * 2.0 * half_tan / ch, 1e-6)
                m = _MIN_HIT_PX * one_px
                t = _ray_aabb(ro, rd, center, hx + m, hy + m, hz + m)
            if t is not None and t > 1e-3:
                candidates.append((t, name, center, geom_data, shape))

        candidates.sort(key=lambda x: x[0])

        if candidates:
            dbg = "  ".join(f"{n}:t={t:.3f}" for t, n, *_ in candidates)
        else:
            dbg = "no hits"
        print(f"[pick] click=({sx:.0f},{sy:.0f})  {dbg}", flush=True)

        if not candidates:
            ctr = np.array(cam.center, dtype=float)
            ref = ctr if plane_pt is None else np.array(plane_pt, dtype=float)
            t_plane = float(np.dot(ref - ro, rd))
            return (ro + rd * t_plane, None) if t_plane > 0 else (None, None)

        # Recompute hit point on the TRUE (un-expanded) geometry for the winner
        _, best_obj, best_center, best_geom, best_shape = candidates[0]
        if best_shape == "cyl":
            t_true = _ray_zcylinder(ro, rd, best_center, best_geom, float(CYL_HALF_H))
        else:
            hx, hy, hz = best_geom
            t_true = _ray_aabb(ro, rd, best_center, hx, hy, hz)
        t_final = t_true if (t_true is not None and t_true > 1e-3) else candidates[0][0]
        return ro + rd * t_final, best_obj

    _debug_dot = visuals.Markers(parent=view.scene)
    _debug_dot.visible = False

    # ── Keyboard: A / B to select, X / Y / Z to lock axis, D to deselect ──────

    @canvas.events.key_press.connect
    def _on_key_press(event):
        k = event.key.name.lower() if hasattr(event.key, 'name') else str(event.key).lower()
        # D = deselect; X/Y/Z = axis lock while held (for precise axis-aligned drag)
        if k == 'd':
            _sel["who"] = None;  _sel["axis_lock"] = None
        elif k in ('x', 'y', 'z') and _sel["who"] is not None:
            _sel["axis_lock"] = k.upper()
        _refresh_gizmos()

    @canvas.events.key_release.connect
    def _on_key_release(event):
        k = event.key.name.lower() if hasattr(event.key, 'name') else str(event.key).lower()
        if k in ('x', 'y', 'z') and _sel["axis_lock"] == k.upper():
            _sel["axis_lock"] = None
            _refresh_gizmos()

    # ── Mouse: click to project + spawn dot; select + drag ───────────────────────

    def _on_sel_press(event):
        if event.button != 1:
            return

        # Ray-geometry intersection: gives surface hit position AND object name
        hit, geom_obj = _pick_world_pos(event.pos[0], event.pos[1])
        if hit is not None:
            print(f"[pick] px=({event.pos[0]:.0f},{event.pos[1]:.0f})"
                  f"  world={np.round(hit, 3)}  obj={geom_obj!r}", flush=True)
            _debug_dot.set_data(hit[np.newaxis].astype(np.float32),
                                face_color=(1.0, 1.0, 1.0, 1.0), size=14, edge_width=0)
            _debug_dot.visible = True

        if geom_obj is not None:
            # Hit a scene object → select it
            _sel["who"]       = geom_obj
            _sel["axis_lock"] = None
            _refresh_gizmos()
        else:
            # Missed everything → deselect, let camera orbit
            _sel["who"]       = None
            _sel["axis_lock"] = None
            _refresh_gizmos()
            return   # don't block camera

        # All five scene objects are draggable.  Moving a roll or the guide
        # resets the yarn to the new warped geometry; sensors just reposition.
        if _sel["who"] not in ("sensor_a", "sensor_b", "cyl", "roll_a", "roll_b"):
            event._blocked = True
            return
        _sel["dragging"]         = True
        _sel["drag_start_world"] = _object_center(_sel["who"]).copy()
        _sel["drag_start_mouse"] = np.array(event.pos[:2], dtype=float)
        event._blocked = True

    def _on_sel_move(event):
        if not _sel["dragging"] or _sel["who"] is None:
            return
        d_screen = np.array(event.pos[:2], dtype=float) - _sel["drag_start_mouse"]
        world_delta = _screen_delta_to_world(d_screen)
        lock = _sel["axis_lock"]
        if lock is not None:
            ax_dir      = _AX_DIRS[lock]
            world_delta = ax_dir * float(np.dot(world_delta, ax_dir))
        new_pos = _sel["drag_start_world"] + world_delta

        who = _sel["who"]
        x, y, z = float(new_pos[0]), float(new_pos[1]), float(new_pos[2])
        if who == "sensor_a":
            state["sensor_a_x"] = x; state["sensor_a_y"] = y; state["sensor_a_z"] = z
            _sphere_centers[0]  = new_pos
            rebuild_scA()
        elif who == "sensor_b":
            state["sensor_b_x"] = x; state["sensor_b_y"] = y; state["sensor_b_z"] = z
            _sphere_centers[1]  = new_pos
            rebuild_scB()
        elif who == "cyl":
            state["cyl_x"]  = x; state["cyl_y"]  = y; state["cyl_z"]  = z
            rebuild_guide()
            rebuild_detect_vol()
        elif who == "roll_a":
            state["roll_a_x"] = x; state["roll_a_y"] = y; state["roll_a_z"] = z
            rebuild_roll_a()
        elif who == "roll_b":
            state["roll_b_x"] = x; state["roll_b_y"] = y; state["roll_b_z"] = z
            rebuild_roll_b()

        _update_sensor_visuals()
        _refresh_gizmos()
        event._blocked = True

    def _on_sel_release(event):
        if _sel["dragging"]:
            _sel["dragging"] = False
            event._blocked   = True

    canvas.events.mouse_press.connect(_on_sel_press,    position='first')
    canvas.events.mouse_move.connect(_on_sel_move,      position='first')
    canvas.events.mouse_release.connect(_on_sel_release, position='first')

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
                        rebuild_detect_vol()
                    elif key == "cyl_detect_r":
                        rebuild_detect_vol()
                    elif key == "cyl_detect_show":
                        cyl_detect_vis.visible = bool(int(value))
                    elif key in ("sensor_a_x", "sensor_a_y", "sensor_a_z", "sensor_a_cyl_r"):
                        rebuild_scA()
                    elif key == "sensor_a_cyl_enabled":
                        scA_vis.visible = bool(int(value))
                        _graph[0] = None
                    elif key in ("sensor_b_x", "sensor_b_y", "sensor_b_z", "sensor_b_cyl_r"):
                        rebuild_scB()
                    elif key == "sensor_b_cyl_enabled":
                        scB_vis.visible = bool(int(value))
                        _graph[0] = None
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
        _update_sensor_visuals()

        # ── Roll A tension servo (PI on T_a) ─────────────────────────────────
        servo_on = bool(int(state.get("roll_a_servo_on", 0)))
        if servo_on and running[0]:
            now = time.perf_counter()
            dt_ctrl = (now - _servo_t_prev[0]) if _servo_t_prev[0] is not None else (1.0 / 60.0)
            _servo_t_prev[0] = now
            dt_ctrl = max(1e-4, min(dt_ctrl, 0.1))   # clamp to sane range

            setp   = float(state.get("roll_a_tension_setpoint", 5.0))
            kp     = float(state.get("roll_a_kp", 1.0))
            ki     = float(state.get("roll_a_ki", 0.0))
            T_a    = float(shared[0])
            err    = setp - T_a

            # Anti-windup: clamp integral so |ki · I| ≤ omega_max.
            omega_max = 200.0
            _servo_integral[0] += err * dt_ctrl
            if ki > 1e-9:
                lim = omega_max / ki
                if _servo_integral[0] >  lim: _servo_integral[0] =  lim
                if _servo_integral[0] < -lim: _servo_integral[0] = -lim

            omega_cmd = kp * err + ki * _servo_integral[0]
            if   omega_cmd >  omega_max: omega_cmd =  omega_max
            elif omega_cmd < -omega_max: omega_cmd = -omega_max
            _servo_omega_cmd[0] = omega_cmd
            omega_cmd_wp.assign(wp.array([omega_cmd], dtype=float, device=device))
        else:
            _servo_t_prev[0] = None
            # Reset integral while servo is off so re-engagement starts clean.
            _servo_integral[0] = 0.0
            _servo_omega_cmd[0] = 0.0

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
        who  = _sel["who"]
        lock = _sel["axis_lock"]
        if who is not None:
            pt       = _sphere_centers[0] if who == "A" else _sphere_centers[1]
            ax_hint  = f" | holding {lock} axis" if lock else " | hold X/Y/Z to constrain axis"
            drag_line = (
                f"[Sensor {who} selected{ax_hint}]  "
                f"X={pt[0]:+.3f}  Y={pt[1]:+.3f}  Z={pt[2]:+.3f}  — drag to move | D or A/B again to deselect"
            )
        else:
            drag_line = "Press A or B to select sensor | D to deselect | orbit: left-drag"
        if servo_on:
            servo_line = (
                f"SERVO: ON  setpt={float(state.get('roll_a_tension_setpoint', 0)):.2f}cN  "
                f"err={(float(state.get('roll_a_tension_setpoint', 0)) - _T_a):+.2f}cN  "
                f"ωcmd={_servo_omega_cmd[0]:+.1f} rad/s\n"
            )
        else:
            servo_line = "SERVO: OFF (passive flywheel)\n"
        hud.text = (
            f"N={N}  seg={config.REST_LEN*1000:.1f}mm  r={state['ogc_r']:.3f}  "
            f"substeps={config.SUBSTEPS}  iter={config.CONSTRAINT_ITER}\n"
            f"pull={state['pull_speed']:+.2f} m/s  "
            f"ωA={_omega_a:+.1f} rad/s  θA={np.degrees(_angle_a):.0f}°\n"
            f"[{device}|{graph_mode}]  frame {frame[0]:05d}  {status}  "
            f"t={sim_time[0]:.2f}s  step={_frame_ms[0]:.1f}ms\n"
            f"T_A={_T_a:.2f}cN  T_B={_T_b:.2f}cN  "
            f"θ={_theta:.1f}°  pred={_pred:.2f}cN  resid={_resid:.3f}\n"
            f"{servo_line}"
            f"{drag_line}"
        )
        canvas.update()

    timer = app.Timer(interval=1.0 / 60.0, connect=on_timer, start=True)
    print("[sim_worker] entering vispy event loop", flush=True)
    app.run()
    del timer


# ── Tkinter control panel (parent process) ────────────────────────────────────

def run_ui(cmd_queue, shared, dbg_shared):
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
    param_ranges    = {}   # key → (min_var, max_var, apply_fn)  for editable_range sliders

    def add_slider(label, key, from_, to_, default, is_int=False, fmt="{:.3f}",
                   editable_range=False):
        frm = ttk.Frame(scroll_frm)
        frm.pack(fill="x", padx=8, pady=3)
        ttk.Label(frm, text=label, width=22).pack(side="left")

        val_var   = tk.DoubleVar(value=default)
        init_disp = f"{int(default)} / {int(to_)}" if is_int else fmt.format(default)
        disp_var  = tk.StringVar(value=init_disp)
        ttk.Label(frm, textvariable=disp_var, width=10).pack(side="right")

        # Mutable range so inner closures can update it.
        _range = [from_, to_]

        def on_change(v):
            if is_int:
                iv = int(round(float(v)))
                val_var.set(iv)
                disp_var.set(f"{iv} / {int(_range[1])}")
                cmd_queue.put(("param", key, iv))
            else:
                fv = float(v)
                val_var.set(fv)
                disp_var.set(fmt.format(fv))
                cmd_queue.put(("param", key, fv))

        scale = ttk.Scale(frm, from_=_range[0], to=_range[1], variable=val_var,
                          orient="horizontal", command=on_change)
        scale.pack(side="left", fill="x", expand=True, padx=6)

        if editable_range:
            # Second row: [min entry] [max entry] with labels
            range_frm = ttk.Frame(scroll_frm)
            range_frm.pack(fill="x", padx=8, pady=(0, 2))
            ttk.Label(range_frm, text="  range:", width=9,
                      foreground="#888888").pack(side="left")

            min_var = tk.StringVar(value=str(from_))
            max_var = tk.StringVar(value=str(to_))

            def _apply_range(*_):
                try:
                    new_lo = (int if is_int else float)(min_var.get())
                    new_hi = (int if is_int else float)(max_var.get())
                    if new_lo >= new_hi:
                        return
                    _range[0] = new_lo;  _range[1] = new_hi
                    scale.configure(from_=new_lo, to=new_hi)
                    # Re-clamp current value if outside new range.
                    cur = val_var.get()
                    if cur < new_lo:
                        val_var.set(new_lo);  on_change(new_lo)
                    elif cur > new_hi:
                        val_var.set(new_hi);  on_change(new_hi)
                except ValueError:
                    pass

            ttk.Label(range_frm, text="min", foreground="#888888").pack(side="left")
            min_e = ttk.Entry(range_frm, textvariable=min_var, width=8)
            min_e.pack(side="left", padx=(2, 8))
            min_e.bind("<Return>", _apply_range)
            min_e.bind("<FocusOut>", _apply_range)

            ttk.Label(range_frm, text="max", foreground="#888888").pack(side="left")
            max_e = ttk.Entry(range_frm, textvariable=max_var, width=8)
            max_e.pack(side="left", padx=2)
            max_e.bind("<Return>", _apply_range)
            max_e.bind("<FocusOut>", _apply_range)

            # Expose this slider's range so save/load can persist it.
            param_ranges[key] = (min_var, max_var, _apply_range)

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
    add_slider("Friction μ_kinetic",   "mu_kinetic",   0.0,   1.0,  DEFAULTS["mu_kinetic"], editable_range=True)
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
    section("Tension sensor A — upstream (yellow)")
    add_slider("Sensor A  X", "sensor_a_x", -3.0, 3.0, DEFAULTS["sensor_a_x"], fmt="{:+.3f}")
    add_slider("Sensor A  Y", "sensor_a_y", -3.0, 3.0, DEFAULTS["sensor_a_y"], fmt="{:+.3f}")
    add_slider("Sensor A  Z", "sensor_a_z", -3.0, 3.0, DEFAULTS["sensor_a_z"], fmt="{:+.3f}")
    add_slider("Sensor A  half-extent X (m)", "sensor_a_hx", 0.001, 0.5, DEFAULTS["sensor_a_hx"], fmt="{:.3f}")
    add_slider("Sensor A  half-extent Y (m)", "sensor_a_hy", 0.001, 0.5, DEFAULTS["sensor_a_hy"], fmt="{:.3f}")
    add_slider("Sensor A  half-extent Z (m)", "sensor_a_hz", 0.001, 0.5, DEFAULTS["sensor_a_hz"], fmt="{:.3f}")
    add_slider("Sensor A  opacity",           "sensor_a_alpha", 0.0, 1.0, DEFAULTS["sensor_a_alpha"], fmt="{:.2f}")
    _scA_en_var = tk.IntVar(value=DEFAULTS["sensor_a_cyl_enabled"])
    _scA_en_frm = ttk.Frame(scroll_frm)
    _scA_en_frm.pack(fill="x", padx=8, pady=3)
    def _on_scA_toggle():
        cmd_queue.put(("param", "sensor_a_cyl_enabled", _scA_en_var.get()))
    ttk.Checkbutton(_scA_en_frm, text="Enable guide cylinder A (frictionless)",
                    variable=_scA_en_var, command=_on_scA_toggle).pack(side="left")
    param_vars["sensor_a_cyl_enabled"]      = _scA_en_var
    param_callbacks["sensor_a_cyl_enabled"] = lambda v: (
        _scA_en_var.set(int(v)), cmd_queue.put(("param", "sensor_a_cyl_enabled", int(v)))
    )
    add_slider("Sensor A  cylinder radius (m)", "sensor_a_cyl_r", 0.005, 0.3, DEFAULTS["sensor_a_cyl_r"], fmt="{:.3f}")

    section("Tension sensor B — downstream (green)")
    add_slider("Sensor B  X", "sensor_b_x", -3.0, 3.0, DEFAULTS["sensor_b_x"], fmt="{:+.3f}")
    add_slider("Sensor B  Y", "sensor_b_y", -3.0, 3.0, DEFAULTS["sensor_b_y"], fmt="{:+.3f}")
    add_slider("Sensor B  Z", "sensor_b_z", -3.0, 3.0, DEFAULTS["sensor_b_z"], fmt="{:+.3f}")
    add_slider("Sensor B  half-extent X (m)", "sensor_b_hx", 0.001, 0.5, DEFAULTS["sensor_b_hx"], fmt="{:.3f}")
    add_slider("Sensor B  half-extent Y (m)", "sensor_b_hy", 0.001, 0.5, DEFAULTS["sensor_b_hy"], fmt="{:.3f}")
    add_slider("Sensor B  half-extent Z (m)", "sensor_b_hz", 0.001, 0.5, DEFAULTS["sensor_b_hz"], fmt="{:.3f}")
    add_slider("Sensor B  opacity",           "sensor_b_alpha", 0.0, 1.0, DEFAULTS["sensor_b_alpha"], fmt="{:.2f}")
    _scB_en_var = tk.IntVar(value=DEFAULTS["sensor_b_cyl_enabled"])
    _scB_en_frm = ttk.Frame(scroll_frm)
    _scB_en_frm.pack(fill="x", padx=8, pady=3)
    def _on_scB_toggle():
        cmd_queue.put(("param", "sensor_b_cyl_enabled", _scB_en_var.get()))
    ttk.Checkbutton(_scB_en_frm, text="Enable guide cylinder B (frictionless)",
                    variable=_scB_en_var, command=_on_scB_toggle).pack(side="left")
    param_vars["sensor_b_cyl_enabled"]      = _scB_en_var
    param_callbacks["sensor_b_cyl_enabled"] = lambda v: (
        _scB_en_var.set(int(v)), cmd_queue.put(("param", "sensor_b_cyl_enabled", int(v)))
    )
    add_slider("Sensor B  cylinder radius (m)", "sensor_b_cyl_r", 0.005, 0.3, DEFAULTS["sensor_b_cyl_r"], fmt="{:.3f}")

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

    section("Roll A — tension servo (PI on T_A)")
    _servo_on_var = tk.IntVar(value=DEFAULTS["roll_a_servo_on"])
    _servo_on_frm = ttk.Frame(scroll_frm)
    _servo_on_frm.pack(fill="x", padx=8, pady=3)
    def _on_servo_toggle():
        cmd_queue.put(("param", "roll_a_servo_on", _servo_on_var.get()))
    ttk.Checkbutton(_servo_on_frm, text="Enable servo (overrides passive flywheel)",
                    variable=_servo_on_var, command=_on_servo_toggle).pack(side="left")
    param_vars["roll_a_servo_on"]      = _servo_on_var
    param_callbacks["roll_a_servo_on"] = lambda v: (
        _servo_on_var.set(int(v)), cmd_queue.put(("param", "roll_a_servo_on", int(v)))
    )
    add_slider("Tension setpoint (cN)", "roll_a_tension_setpoint", 0.0, 50.0,
               DEFAULTS["roll_a_tension_setpoint"], fmt="{:.2f}", editable_range=True)
    add_slider("Servo kp (rad/s per cN)", "roll_a_kp", -20.0, 20.0,
               DEFAULTS["roll_a_kp"], fmt="{:+.3f}", editable_range=True)
    add_slider("Servo ki (rad/s per cN·s)", "roll_a_ki", -20.0, 20.0,
               DEFAULTS["roll_a_ki"], fmt="{:+.3f}", editable_range=True)

    section("Roll B — pulling roll")
    add_slider("Roll B  X",      "roll_b_x",      -3.0,  3.0, DEFAULTS["roll_b_x"],      fmt="{:+.3f}")
    add_slider("Roll B  Y",      "roll_b_y",      -3.0,  3.0, DEFAULTS["roll_b_y"],      fmt="{:+.3f}")
    add_slider("Roll B  Z",      "roll_b_z",      -3.0,  3.0, DEFAULTS["roll_b_z"],      fmt="{:+.3f}")
    add_slider("Roll B  radius", "roll_b_radius",  0.02, 0.5, DEFAULTS["roll_b_radius"])
    add_slider("Pull speed (m/s)", "pull_speed",  -5.0,  5.0, DEFAULTS["pull_speed"],    fmt="{:+.3f}", editable_range=True)

    section("Guide cylinder")
    add_slider("Guide X",      "cyl_x",      -3.0,  3.0, DEFAULTS["cyl_x"],      fmt="{:+.3f}", editable_range=True)
    add_slider("Guide Y",      "cyl_y",      -3.0,  3.0, DEFAULTS["cyl_y"],      fmt="{:+.3f}", editable_range=True)
    add_slider("Guide Z",      "cyl_z",      -3.0,  3.0, DEFAULTS["cyl_z"],      fmt="{:+.3f}", editable_range=True)
    add_slider("Guide radius", "cyl_radius",  0.02, 0.5, DEFAULTS["cyl_radius"])
    add_slider("Wrap-detect radius (m)", "cyl_detect_r", 0.05, 1.0,
               DEFAULTS["cyl_detect_r"], fmt="{:.3f}")
    _cyl_det_show_var = tk.IntVar(value=DEFAULTS["cyl_detect_show"])
    _cyl_det_show_frm = ttk.Frame(scroll_frm)
    _cyl_det_show_frm.pack(fill="x", padx=8, pady=3)
    def _on_cyl_det_show_toggle():
        cmd_queue.put(("param", "cyl_detect_show", _cyl_det_show_var.get()))
    ttk.Checkbutton(_cyl_det_show_frm, text="Show wrap-detect volume",
                    variable=_cyl_det_show_var,
                    command=_on_cyl_det_show_toggle).pack(side="left")
    param_vars["cyl_detect_show"]      = _cyl_det_show_var
    param_callbacks["cyl_detect_show"] = lambda v: (
        _cyl_det_show_var.set(int(v)),
        cmd_queue.put(("param", "cyl_detect_show", int(v)))
    )

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
        # Persist custom slider ranges so widened sliders survive a save/load
        # round trip (otherwise editable_range sliders snap back to defaults
        # on load and silently clamp values outside the default range).
        ranges = {}
        for key, (min_var, max_var, _) in param_ranges.items():
            try:
                ranges[key] = [float(min_var.get()), float(max_var.get())]
            except ValueError:
                pass
        if ranges:
            data["_ranges"] = ranges
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
        # Restore custom slider ranges FIRST so subsequent value loads don't
        # get clamped to a narrower default range.
        ranges = data.get("_ranges", {}) or {}
        for key, lo_hi in ranges.items():
            if key in param_ranges and isinstance(lo_hi, (list, tuple)) and len(lo_hi) == 2:
                min_var, max_var, apply_fn = param_ranges[key]
                min_var.set(str(lo_hi[0]))
                max_var.set(str(lo_hi[1]))
                apply_fn()
        for key, value in data.items():
            if key == "_ranges":
                continue
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

    # Auto-load default params at startup so all saved preferences
    # (including toggles) are applied without the user needing to click Load.
    _autoload_path = os.path.join(_SCRIPT_DIR, "params-main-cotton-ceramic-2.json")
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
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    _GRAPH_HISTORY = 15      # seconds of rolling history to show
    _GRAPH_HZ      = 20      # update rate (Hz)
    _BUF_LEN       = _GRAPH_HISTORY * _GRAPH_HZ * 4   # ample buffer

    def _make_bufs():
        return (collections.deque(maxlen=_BUF_LEN),
                collections.deque(maxlen=_BUF_LEN),
                collections.deque(maxlen=_BUF_LEN))

    _t_buf, _Ta_buf, _Tb_buf = _make_bufs()

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

    # Matplotlib navigation toolbar — Home, Pan, Zoom-rect, Save, etc.
    tool_frm = ttk.Frame(graph_win)
    tool_frm.pack(fill="x")
    graph_toolbar = NavigationToolbar2Tk(graph_canvas, tool_frm, pack_toolbar=False)
    graph_toolbar.update()
    graph_toolbar.pack(side="left")

    # Auto-fit toggle — when off, the rolling auto-rescale is suspended so the
    # user can inspect a frozen region.  Toggled OFF automatically the moment
    # they zoom or pan; toggled back ON by clicking "Auto-fit" or "Home".
    _auto_fit = tk.BooleanVar(value=True)
    _auto_fit_chk = ttk.Checkbutton(tool_frm, text="Auto-fit",
                                    variable=_auto_fit)
    _auto_fit_chk.pack(side="left", padx=12)

    # Text panel below the plot for Capstan breakdown
    info_frm = ttk.Frame(graph_win)
    info_frm.pack(fill="x", padx=8, pady=4)
    _info_var = tk.StringVar(value="Waiting for simulation data...")
    ttk.Label(info_frm, textvariable=_info_var, font=("Courier", 9),
              foreground="white", background="#1e1e1e").pack(anchor="w")
    info_frm.configure(style="Dark.TFrame")

    _last_sim_t  = [-1.0]
    _graph_alive = [True]

    # When _update_graph adjusts the axes we set this flag so the limit-changed
    # callbacks don't mistake it for a user action and disable auto-fit.
    _programmatic_lim = [False]

    def _on_user_axes_change(_ax):
        if not _programmatic_lim[0]:
            _auto_fit.set(False)

    ax.callbacks.connect("xlim_changed", _on_user_axes_change)
    ax.callbacks.connect("ylim_changed", _on_user_axes_change)

    # Scroll-wheel zoom centred on the cursor position.  Ctrl+scroll = X only,
    # Shift+scroll = Y only, otherwise both axes scale together.
    def _on_scroll(event):
        if event.inaxes is not ax:
            return
        scale = 1 / 1.2 if event.button == "up" else 1.2
        x_only = bool(event.key and "control" in event.key)
        y_only = bool(event.key and "shift"   in event.key)
        xlo, xhi = ax.get_xlim();  ylo, yhi = ax.get_ylim()
        cx, cy = event.xdata, event.ydata
        if not y_only:
            ax.set_xlim(cx - (cx - xlo) * scale, cx + (xhi - cx) * scale)
        if not x_only:
            ax.set_ylim(cy - (cy - ylo) * scale, cy + (yhi - cy) * scale)
        _auto_fit.set(False)
        graph_canvas.draw_idle()

    graph_canvas.mpl_connect("scroll_event", _on_scroll)

    def _clear_bufs():
        _t_buf.clear();  _Ta_buf.clear();  _Tb_buf.clear()
        line_a.set_data([], []);  line_b.set_data([], [])
        graph_canvas.draw_idle()

    def _update_graph():
        if not _graph_alive[0]:
            return

        sim_t = shared[5]
        # Detect reset: sim_time jumped back (worker reset to 0).
        if sim_t < _last_sim_t[0] - 0.1:
            _clear_bufs()
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
            if _auto_fit.get():
                _programmatic_lim[0] = True
                ax.set_xlim(max(0.0, t_lo), max(t_now, _GRAPH_HISTORY))
                all_vals = ta_arr + tb_arr
                v_min = min(all_vals);  v_max = max(all_vals)
                pad   = max(0.5, (v_max - v_min) * 0.15)
                ax.set_ylim(v_min - pad, v_max + pad)
                _programmatic_lim[0] = False
            graph_canvas.draw_idle()

        T_a    = shared[0];  T_b   = shared[1]
        theta  = shared[2];  pred  = shared[3];  resid = shared[4]
        n_ctc  = int(shared[6])
        mu_k   = float(param_vars.get("mu_kinetic", tk.DoubleVar(value=0.0)).get())
        theta_r = math.radians(theta)
        _info_var.set(
            f"T_A = {T_a:7.3f} cN  (upstream,  yellow)      "
            f"T_B = {T_b:7.3f} cN  (downstream, green)\n"
            f"Wrap angle θ = {theta:.1f}° ({n_ctc} pts on guide)    "
            f"μ_k = {mu_k:.4f}    e^(μ_k·θ) = {math.exp(mu_k * theta_r):.4f}\n"
            f"Measured ratio T_B/T_A = {(T_b/T_a if T_a > 1e-6 else 0):.4f}    "
            f"Capstan pred T_B = {pred:.3f} cN    Residual = {resid:.4f}"
        )

        root.after(1000 // _GRAPH_HZ, _update_graph)

    root.after(500, _update_graph)   # start after half a second

    # ── Wrap-angle debug top-down view ────────────────────────────────────────
    # A live 2D rendering of what _wrap_angle_contacts is doing:
    #   • dashed grey circle = search disk (radius = cyl_detect_r)
    #   • dashed dark-green pair = arc-band edges (r_contact ± arc_tol)
    #   • solid green ring = contact circle (r_cyl + ogc_r)
    #   • olive disc = cylinder cross-section (r_cyl)
    #   • dots = particles in the search region, coloured by classification:
    #       grey   = in search disk only (rejected by longest-run filter)
    #       yellow = in the longest contiguous yarn run
    #       green  = passed arc-band filter — these are the particles used
    #                in the θ summation
    #   • bright green polyline = the arc, in yarn-index order
    #   • pink/blue rays + rings = refined in/out tangent points (the
    #                              true boundary of the wrap arc)
    WRAP_BG = "#101015"
    wrap_win = tk.Toplevel(root)
    wrap_win.title("Wrap-angle debug — top-down view")
    wrap_win.geometry("560x640")
    wrap_win.protocol("WM_DELETE_WINDOW", lambda: None)
    wrap_win.configure(bg=WRAP_BG)

    wrap_canvas = tk.Canvas(wrap_win, bg=WRAP_BG, highlightthickness=0)
    wrap_canvas.pack(fill="both", expand=True)

    wrap_info_var = tk.StringVar(value="Waiting for sim data...")
    tk.Label(wrap_win, textvariable=wrap_info_var, font=("Courier", 9),
             fg="white", bg=WRAP_BG, justify="left", anchor="w"
             ).pack(fill="x", padx=8, pady=(2, 0))

    legend_frm = tk.Frame(wrap_win, bg=WRAP_BG)
    legend_frm.pack(fill="x", padx=8, pady=(2, 6))
    for txt, col in (("● search",       "#888899"),
                     ("● longest",      "#ffcc44"),
                     ("● arc",          "#33ff88"),
                     ("│ Δθ ticks",     "#22aa55"),
                     ("◜ swept θ →",    "#ff44dd"),
                     ("○ in",           "#ff5577"),
                     ("○ out",          "#55aaff")):
        tk.Label(legend_frm, text=txt, fg=col, bg=WRAP_BG,
                 font=("Courier", 9)).pack(side="left", padx=4)

    # View rotation — lets the user spin the whole top-down picture so a
    # known reference (typically the refined-in tangent) aligns to +X.  Then
    # checking "is this 90°?" becomes "is the other tangent vertical?".
    _view_rot_deg = tk.DoubleVar(value=0.0)
    rot_frm = tk.Frame(wrap_win, bg=WRAP_BG)
    rot_frm.pack(fill="x", padx=8, pady=(0, 6))
    tk.Label(rot_frm, text="View rotation°", fg="white", bg=WRAP_BG,
             font=("Courier", 9)).pack(side="left")
    ttk.Scale(rot_frm, from_=-180.0, to=180.0, variable=_view_rot_deg,
              orient="horizontal").pack(side="left", fill="x",
                                         expand=True, padx=6)
    _view_rot_lbl = tk.Label(rot_frm, text="+0°", fg="white", bg=WRAP_BG,
                             font=("Courier", 9), width=6)
    _view_rot_lbl.pack(side="left")

    def _snap_in_to_zero():
        # Rotate so the refined "in" tangent lies on the +X axis.
        rin_x = float(dbg_shared[10]); rin_y = float(dbg_shared[11])
        if math.isnan(rin_x) or math.isnan(rin_y):
            return
        ang = math.degrees(math.atan2(rin_y, rin_x))
        _view_rot_deg.set(-ang)

    def _reset_rotation():
        _view_rot_deg.set(0.0)

    tk.Button(rot_frm, text="in→0°", command=_snap_in_to_zero,
              font=("Courier", 8)).pack(side="left", padx=(4, 0))
    tk.Button(rot_frm, text="reset", command=_reset_rotation,
              font=("Courier", 8)).pack(side="left", padx=(2, 0))

    _wrap_alive = [True]

    def _update_wrap_debug():
        if not _wrap_alive[0]:
            return
        cv = wrap_canvas
        cv.delete("all")
        W = cv.winfo_width();  H = cv.winfo_height()
        if W < 60 or H < 60:
            root.after(150, _update_wrap_debug)
            return

        n_search    = int(dbg_shared[0])
        r_cyl       = float(dbg_shared[3])
        r_contact   = float(dbg_shared[4])
        arc_tol     = float(dbg_shared[5])
        rd          = float(dbg_shared[6])
        theta       = float(dbg_shared[7])
        n_arc       = int(dbg_shared[8])
        longest_len = int(dbg_shared[9])
        rin_x       = float(dbg_shared[10])
        rin_y       = float(dbg_shared[11])
        rout_x      = float(dbg_shared[12])
        rout_y      = float(dbg_shared[13])
        signed_th   = float(dbg_shared[15])

        ext = max(rd * 1.15, r_contact * 1.5, 0.05)
        margin = 24
        sz = min(W, H) - 2 * margin
        if sz <= 0:
            root.after(150, _update_wrap_debug)
            return
        scale = sz / (2.0 * ext)
        ox    = W / 2.0
        oy    = H / 2.0

        rot_deg = float(_view_rot_deg.get())
        _view_rot_lbl.config(text=f"{rot_deg:+4.0f}°")
        ca = math.cos(math.radians(rot_deg))
        sa = math.sin(math.radians(rot_deg))

        def to_cv(x, y):
            # Rotate (world) by rot_deg CCW, then map to canvas with Y-flip.
            xr = x * ca - y * sa
            yr = x * sa + y * ca
            return ox + xr * scale, oy - yr * scale

        def draw_circle(r, **kw):
            # Concentric on origin — bounding box is rotation-invariant, so
            # bypass to_cv (avoids drawing a rotated rectangle's bbox).
            cv.create_oval(ox - r * scale, oy - r * scale,
                           ox + r * scale, oy + r * scale, **kw)

        if rd > 0:
            draw_circle(rd, outline="#3a4a5a", width=1, dash=(4, 4))
        if r_contact > 0 and arc_tol > 0:
            draw_circle(r_contact - arc_tol, outline="#225544", width=1, dash=(2, 3))
            draw_circle(r_contact + arc_tol, outline="#225544", width=1, dash=(2, 3))
        if r_contact > 0:
            draw_circle(r_contact, outline="#33aa55", width=2)
        if r_cyl > 0:
            draw_circle(r_cyl, outline="#aabb55", fill="#262318", width=2)

        ox_c, oy_c = to_cv(0, 0)
        cv.create_line(ox_c - 7, oy_c,     ox_c + 7, oy_c,     fill="#555566")
        cv.create_line(ox_c,     oy_c - 7, ox_c,     oy_c + 7, fill="#555566")

        # Radial measurement lines: from each arc particle to the centroid.
        # Drawn first so they sit behind everything else.  Lets you eyeball
        # angles between any two arc particles directly.
        for k in range(min(n_search, MAX_DEBUG_PARTS)):
            off = DBG_HEADER_LEN + DBG_STRIDE * k
            if int(dbg_shared[off + 3]) != 2:
                continue
            px = float(dbg_shared[off + 0])
            py = float(dbg_shared[off + 1])
            xe, ye = to_cv(px, py)
            cv.create_line(ox_c, oy_c, xe, ye,
                           fill="#2a5a3a", width=1)

        arc_xy_cv = []
        n = min(n_search, MAX_DEBUG_PARTS)
        for k in range(n):
            off = DBG_HEADER_LEN + DBG_STRIDE * k
            px  = float(dbg_shared[off + 0])
            py  = float(dbg_shared[off + 1])
            cls = int(dbg_shared[off + 3])
            cxp, cyp = to_cv(px, py)
            if cls == 2:
                col, rdot = "#33ff88", 4
                arc_xy_cv.append((cxp, cyp))
            elif cls == 1:
                col, rdot = "#ffcc44", 3
            else:
                col, rdot = "#888899", 2
            cv.create_oval(cxp - rdot, cyp - rdot,
                           cxp + rdot, cyp + rdot,
                           fill=col, outline="")

        if len(arc_xy_cv) >= 2:
            flat = [c for pt in arc_xy_cv for c in pt]
            cv.create_line(*flat, fill="#66ffaa", width=2)

        # ── Visualize what the algorithm THINKS the angle is ─────────────
        # The arc is drawn on the contact circle, starting from the refined
        # "in" azimuth, sweeping signed_theta radians (signed = direction
        # the algorithm summed Δθ).  If the algorithm is correct, the arc
        # end should land on the refined "out" boundary marker.  Any
        # mismatch is the bug, visualized.
        start_ang = None
        if not (math.isnan(rin_x) or math.isnan(rin_y)):
            start_ang = math.atan2(rin_y, rin_x)
        else:
            # Fallback: first arc particle (in yarn-index order)
            for k in range(min(n_search, MAX_DEBUG_PARTS)):
                off = DBG_HEADER_LEN + DBG_STRIDE * k
                if int(dbg_shared[off + 3]) == 2:
                    px = float(dbg_shared[off + 0])
                    py = float(dbg_shared[off + 1])
                    if px*px + py*py > 1e-12:
                        start_ang = math.atan2(py, px)
                    break

        if start_ang is not None and abs(signed_th) > 1e-6 and r_contact > 0:
            r_draw = r_contact * 1.04   # slightly outside contact for visibility
            n_samp = max(16, int(abs(math.degrees(signed_th)) / 3.0))
            pts = []
            for i in range(n_samp + 1):
                t = i / n_samp
                a = start_ang + signed_th * t
                px = r_draw * math.cos(a)
                py = r_draw * math.sin(a)
                pts.append(to_cv(px, py))
            flat = [c for pt in pts for c in pt]
            cv.create_line(*flat, fill="#ff44dd", width=4, capstyle="round")

            # θ label near the midpoint of the swept arc
            mid_a = start_ang + signed_th * 0.5
            lx, ly = to_cv(r_draw * 1.22 * math.cos(mid_a),
                           r_draw * 1.22 * math.sin(mid_a))
            cv.create_text(lx, ly,
                           text=f"θ = {math.degrees(abs(signed_th)):.1f}°",
                           fill="#ff66ee",
                           font=("Courier", 11, "bold"))

            # Small arrow head at the arc end to show sweep direction.
            # Tangent direction in canvas space (accounting for Y-flip).
            end_a = start_ang + signed_th
            ex, ey = to_cv(r_draw * math.cos(end_a),
                           r_draw * math.sin(end_a))
            # World CCW tangent at angle a is (-sin a, cos a); after the
            # view rotation by rot_deg and then the canvas Y-flip, this
            # becomes (-sin(a+rot), -cos(a+rot)).  Reverse for CW.
            rot_rad = math.radians(rot_deg)
            tx_cv = -math.sin(end_a + rot_rad)
            ty_cv = -math.cos(end_a + rot_rad)
            if signed_th < 0:
                tx_cv, ty_cv = -tx_cv, -ty_cv
            tang_ang = math.atan2(ty_cv, tx_cv)
            head_len = 9
            for da in (math.radians(150), math.radians(-150)):
                hx = ex + head_len * math.cos(tang_ang + da)
                hy = ey + head_len * math.sin(tang_ang + da)
                cv.create_line(ex, ey, hx, hy, fill="#ff44dd",
                               width=3, capstyle="round")

        # Radial ticks at each arc particle's azimuth (the discrete sample
        # points the algorithm sums Δθ across, pre-refinement)
        for k in range(min(n_search, MAX_DEBUG_PARTS)):
            off = DBG_HEADER_LEN + DBG_STRIDE * k
            if int(dbg_shared[off + 3]) != 2:
                continue
            px = float(dbg_shared[off + 0])
            py = float(dbg_shared[off + 1])
            r2 = px*px + py*py
            if r2 < 1e-12:
                continue
            inv = 1.0 / math.sqrt(r2)
            ux, uy = px * inv, py * inv     # unit radial in world coords
            r_in  = r_contact * 0.93
            r_out = r_contact * 1.07
            x0, y0 = to_cv(ux * r_in,  uy * r_in)
            x1, y1 = to_cv(ux * r_out, uy * r_out)
            cv.create_line(x0, y0, x1, y1, fill="#22aa55", width=1)

        if not (math.isnan(rin_x) or math.isnan(rin_y)):
            cxp, cyp = to_cv(rin_x, rin_y)
            cv.create_line(ox_c, oy_c, cxp, cyp,
                           fill="#ff5577", width=1, dash=(3, 3))
            cv.create_oval(cxp - 6, cyp - 6, cxp + 6, cyp + 6,
                           outline="#ff5577", width=2)
        if not (math.isnan(rout_x) or math.isnan(rout_y)):
            cxp, cyp = to_cv(rout_x, rout_y)
            cv.create_line(ox_c, oy_c, cxp, cyp,
                           fill="#55aaff", width=1, dash=(3, 3))
            cv.create_oval(cxp - 6, cyp - 6, cxp + 6, cyp + 6,
                           outline="#55aaff", width=2)

        wrap_info_var.set(
            f"θ = {math.degrees(theta):6.1f}°    n_arc = {n_arc}    "
            f"longest_run = {longest_len}    n_search = {n_search}\n"
            f"r_cyl = {r_cyl:.4f}    r_contact = {r_contact:.4f}    "
            f"arc_tol = ±{arc_tol:.4f}    rd = {rd:.3f}"
        )
        root.after(100, _update_wrap_debug)

    root.after(700, _update_wrap_debug)

    def on_close():
        _graph_alive[0] = False
        _wrap_alive[0]  = False
        send("stop")
        plt.close(fig)          # release matplotlib Tk callbacks before destroy
        root.quit()             # exits mainloop(); __main__ finally block cleans up

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
    root.destroy()              # safe to destroy now that mainloop has returned


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # Shared memory: [T_a(cN), T_b(cN), theta_deg, capstan_pred(cN), residual, sim_time]
    shared = mp.Array('d', 7)
    # Wrap-angle debug: header + per-particle classification block (see top of file)
    dbg_shared = mp.Array('d', DBG_ARRAY_LEN)

    cmd_queue = mp.Queue()
    worker = mp.Process(
        target=sim_worker,
        args=(cmd_queue, shared, dbg_shared, _SCRIPT_DIR, DEFAULTS),
    )
    worker.start()

    try:
        run_ui(cmd_queue, shared, dbg_shared)
    finally:
        cmd_queue.put(("stop",))
        worker.join(timeout=5.0)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=2.0)
        if worker.is_alive():
            worker.kill()   # SIGKILL — last resort if CUDA blocks SIGTERM
