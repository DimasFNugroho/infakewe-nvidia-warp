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
    # Physics
    "gravity_y":        -9.81,
    "particle_mass":     0.01,
    "damping":           0.996,
    "stretch_stiff":     1.0,
    "bend_stiff":        0.0,
    "substeps":          20,
    "constraint_iter":   1,
    # OGC contact
    "ogc_r":             0.005,
    "ogc_stiff":         1.0,
    "mu_static":         0.4,
    "mu_kinetic":        0.2,
    "v_max":             20.0,
    # Roll A — feeding roll
    "roll_a_x":         -0.8,
    "roll_a_y":          0.0,
    "roll_a_z":          0.0,
    "roll_a_radius":     0.15,
    "roll_a_mass":       0.5,    # kg — sets rotational inertia of roll A
    "roll_a_bearing_damping": 0.98,   # per-substep drag on omega  [0..1]
    "roll_a_torque_scale":    0.05,   # gain on yarn→roll torque (compensates PBD overestimate)
    # Roll B — pulling roll
    "roll_b_x":          0.8,
    "roll_b_y":          0.0,
    "roll_b_z":          0.0,
    "roll_b_radius":     0.15,
    "pull_speed":        0.0,    # m/s at roll B surface; negative = reverse
    # Visualisation
    "heatmap_mode":      0,     # 0 = stripe colours, 1 = stretch heatmap
    "heatmap_max_strain": 0.05, # strain value that maps to full red  [0..1]
    # Guide cylinder
    "cyl_x":             0.0,
    "cyl_y":            -0.4,
    "cyl_z":             0.0,
    "cyl_radius":        0.08,
}


# ── Simulation worker process ─────────────────────────────────────────────────

def sim_worker(cmd_queue, script_dir: str, defaults: dict):
    """Run Warp + OGC (3 obstacles) + vispy in a dedicated process."""
    sys.path.insert(0, os.path.join(script_dir, ".."))
    sys.path.insert(0, script_dir)

    import queue as py_queue
    import numpy as np
    import warp as wp
    from vispy import app, scene

    from vispy.scene import visuals

    import config
    # ── Override yarn geometry for the roll-to-roll scenario ─────────────────
    config.NUM_PARTICLES = 120
    config.YARN_LENGTH   = 7.0
    config.REST_LEN      = config.YARN_LENGTH / (config.NUM_PARTICLES - 1)

    from ogc.mesh       import build_cylinder, mesh_for_render
    from ogc.algorithm1 import ObstacleGPU, VFContacts, detect_vertex_facet
    from ogc.algorithm2 import EEContacts, detect_edge_edge
    from ogc.algorithm4 import (project_vf, project_ee,
                                 apply_vf_friction, apply_ee_friction,
                                 damp_normal_velocity, clamp_velocity,
                                 roll_a_torque_step, roll_b_motor_step,
                                 set_particle)
    from kernels import (kernel_integrate,
                         kernel_stretch_even, kernel_stretch_odd,
                         kernel_bend, kernel_update_velocity)

    # ── Scene / rendering constants ───────────────────────────────────────────
    CYL_HALF_H = 1.5
    CYL_N_SEGS = 48
    N          = config.NUM_PARTICLES

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

    def make_initial_positions() -> np.ndarray:
        """Helical winding on Roll A, then a nearly-taut free span to Roll B.

        n_free is computed from the straight-line gap distance so that free-span
        particles start at roughly REST_LEN spacing (taut).  All remaining
        particles go to the wound section, giving more wraps and a denser
        winding appearance.
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

        # Angular step and small Z drift for helix separation between wraps.
        dtheta = config.REST_LEN / orbit_r_a
        dz     = config.REST_LEN * 0.15

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

        return np.array(positions[:N], dtype=np.float32)

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
    pos_np = make_initial_positions()

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

    # List-of-lists so individual obstacles can be hot-swapped.
    contacts = [
        [obs_a,   vf_a,   ee_a],
        [obs_b,   vf_b,   ee_b],
        [obs_mid, vf_mid, ee_mid],
    ]

    def apply_state():
        config.GRAVITY         = wp.vec3(0.0, float(state["gravity_y"]), 0.0)
        config.DAMPING         = float(state["damping"])
        config.STRETCH_STIFF   = float(state["stretch_stiff"])
        config.BEND_STIFF      = float(state["bend_stiff"])
        config.SUBSTEPS        = int(state["substeps"])
        config.CONSTRAINT_ITER = int(state["constraint_iter"])

    apply_state()

    def sim_reset():
        angle_b[0] = init_angle_b()
        angle_a[0] = init_angle_a()
        omega_a[0] = 0.0
        angle_a_wp.assign(wp.array([angle_a[0]], dtype=float, device=device))
        omega_a_wp.assign(wp.array([0.0],        dtype=float, device=device))
        angle_b_wp.assign(wp.array([angle_b[0]], dtype=float, device=device))
        pos0 = make_initial_positions()
        pos_wp.assign(wp.array(pos0,                              dtype=wp.vec3, device=device))
        prev_pos_wp.assign(wp.array(pos0.copy(),                  dtype=wp.vec3, device=device))
        vel_wp.assign(wp.array(np.zeros((N, 3), dtype=np.float32), dtype=wp.vec3, device=device))
        inv_mass_wp.assign(wp.array(make_inv_mass(),               dtype=float,   device=device))
        sim_time[0] = 0.0
        frame[0]    = 0

    def sim_step():
        sub_dt    = config.DT / config.SUBSTEPS
        inv_sdt   = 1.0 / sub_dt
        r         = float(state["ogc_r"])
        stiff     = float(state["ogc_stiff"])
        mu_s      = float(state["mu_static"])
        mu_k      = float(state["mu_kinetic"])
        v_max     = float(state["v_max"])
        zero_wind = wp.vec3(0.0, 0.0, 0.0)

        ax, ay, az  = float(state["roll_a_x"]), float(state["roll_a_y"]), float(state["roll_a_z"])
        ra          = max(float(state["roll_a_radius"]), 1e-6)
        M_a         = max(float(state["roll_a_mass"]), 1e-3)
        center_a    = wp.vec3(ax, ay, az)

        bx, by, bz  = float(state["roll_b_x"]), float(state["roll_b_y"]), float(state["roll_b_z"])
        rb          = max(float(state["roll_b_radius"]), 1e-6)
        center_b    = wp.vec3(bx, by, bz)
        # Particles sit at roll_radius + ogc_r so they rest on the OGC offset
        # surface and the contact kernel never fires at the yarn tips.
        orbit_r_a   = ra + r
        orbit_r_b   = rb + r

        for _ in range(config.SUBSTEPS):
            # ── Roll A: passive rotation from yarn tension (revolute joint) ───
            roll_a_torque_step(
                pos_wp, center_a, ra, orbit_r_a,
                config.REST_LEN, config.STRETCH_STIFF,
                float(state["particle_mass"]), M_a,
                sub_dt,
                float(state["roll_a_bearing_damping"]),
                float(state["roll_a_torque_scale"]),
                200.0,
                angle_a_wp, omega_a_wp, device,
            )

            # ── Roll B: motor-driven at pull_speed, sets pos[N-1] on GPU ─────
            roll_b_motor_step(
                pos_wp, center_b, rb, orbit_r_b,
                float(state["pull_speed"]), sub_dt,
                N - 1, angle_b_wp, device,
            )

            # ── Predict ───────────────────────────────────────────────────────
            wp.launch(kernel_integrate, dim=N, device=device,
                      inputs=[pos_wp, vel_wp, prev_pos_wp, inv_mass_wp,
                              config.GRAVITY, zero_wind, sub_dt, config.DAMPING])

            # ── Contact detection ─────────────────────────────────────────────
            for obs, vf, ee in contacts:
                detect_vertex_facet(pos_wp, obs, vf, r, device)
                detect_edge_edge(pos_wp, yarn_edges_wp, obs, ee, r, device)

            # ── Inner PBD iterations ──────────────────────────────────────────
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

            # ── Friction ──────────────────────────────────────────────────────
            for obs, vf, ee in contacts:
                apply_vf_friction(pos_wp, prev_pos_wp, inv_mass_wp,
                                  vf, r, mu_s, mu_k, device)
                apply_ee_friction(pos_wp, prev_pos_wp, inv_mass_wp,
                                  yarn_edges_wp, ee, r, mu_s, mu_k, device)

            # ── Velocity update ───────────────────────────────────────────────
            wp.launch(kernel_update_velocity, dim=N, device=device,
                      inputs=[pos_wp, prev_pos_wp, vel_wp, inv_mass_wp, inv_sdt])
            for obs, vf, ee in contacts:
                damp_normal_velocity(vel_wp, inv_mass_wp, vf, r, device)
            clamp_velocity(vel_wp, v_max, device)

        sim_time[0] += config.DT

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

    hud = visuals.Text(
        text="", color="white", font_size=10, anchor_x="left", anchor_y="top",
        parent=canvas.scene, pos=(10, 20),
    )

    # ── Obstacle rebuild helpers ───────────────────────────────────────────────

    def rebuild_roll_a():
        new_mesh = _cyl("roll_a_x", "roll_a_y", "roll_a_z", "roll_a_radius")
        contacts[0][0] = ObstacleGPU(new_mesh, device)
        vv, ff = mesh_for_render(new_mesh)
        roll_a_vis.set_data(vertices=vv, faces=ff, color=ROLL_A_COL)
        sim_reset()

    def rebuild_roll_b():
        new_mesh = _cyl("roll_b_x", "roll_b_y", "roll_b_z", "roll_b_radius")
        contacts[1][0] = ObstacleGPU(new_mesh, device)
        vv, ff = mesh_for_render(new_mesh)
        roll_b_vis.set_data(vertices=vv, faces=ff, color=ROLL_B_COL)
        sim_reset()

    def rebuild_guide():
        new_mesh = _cyl("cyl_x", "cyl_y", "cyl_z", "cyl_radius")
        contacts[2][0] = ObstacleGPU(new_mesh, device)
        vv, ff = mesh_for_render(new_mesh)
        cyl_vis.set_data(vertices=vv, faces=ff, color=CYL_COL)

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
                    if key in ("roll_a_x", "roll_a_y", "roll_a_z", "roll_a_radius"):
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

        # Read Roll A state from GPU once per frame (not per substep).
        _omega_a = float(omega_a_wp.numpy()[0])
        _angle_a = float(angle_a_wp.numpy()[0])

        status = "RUN" if running[0] else "PAUSED"
        hud.text = (
            f"[{device}]  frame {frame[0]:05d}  {status}  t={sim_time[0]:.2f}s\n"
            f"pull={state['pull_speed']:+.2f} m/s  "
            f"ωA={_omega_a:+.1f} rad/s  θA={np.degrees(_angle_a):.0f}°\n"
            f"r={state['ogc_r']:.3f}  substeps={config.SUBSTEPS}  "
            f"iter={config.CONSTRAINT_ITER}"
        )
        canvas.update()

    timer = app.Timer(interval=1.0 / 60.0, connect=on_timer, start=True)
    print("[sim_worker] entering vispy event loop", flush=True)
    app.run()
    del timer


# ── Tkinter control panel (parent process) ────────────────────────────────────

def run_ui(cmd_queue):
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
    add_slider("Contact stiffness",    "ogc_stiff",    0.0,   1.0,  DEFAULTS["ogc_stiff"])
    add_slider("Friction μ_static",    "mu_static",    0.0,   1.0,  DEFAULTS["mu_static"])
    add_slider("Friction μ_kinetic",   "mu_kinetic",   0.0,   1.0,  DEFAULTS["mu_kinetic"])
    add_slider("Velocity max (m/s)",   "v_max",        0.0,  100.0, DEFAULTS["v_max"],       fmt="{:.1f}")

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

    ttk.Button(btn_frm, text="Start", command=lambda: send("start")
               ).pack(side="left", expand=True, fill="x", padx=2)
    ttk.Button(btn_frm, text="Pause", command=lambda: send("pause")
               ).pack(side="left", expand=True, fill="x", padx=2)
    ttk.Button(btn_frm, text="Reset", command=lambda: send("reset")
               ).pack(side="left", expand=True, fill="x", padx=2)
    ttk.Button(btn_frm, text="Exit",
               command=lambda: (send("stop"), root.after(150, root.destroy))
               ).pack(side="left", expand=True, fill="x", padx=2)

    def on_close():
        send("stop")
        root.after(150, root.destroy)
    root.protocol("WM_DELETE_WINDOW", on_close)

    root.mainloop()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    cmd_queue = mp.Queue()
    worker = mp.Process(
        target=sim_worker, args=(cmd_queue, _SCRIPT_DIR, DEFAULTS),
    )
    worker.start()

    try:
        run_ui(cmd_queue)
    finally:
        cmd_queue.put(("stop",))
        worker.join(timeout=5.0)
        if worker.is_alive():
            worker.terminate()
