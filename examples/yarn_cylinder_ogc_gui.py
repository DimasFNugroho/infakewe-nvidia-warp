"""examples/yarn_cylinder_ogc_gui.py — OGC yarn-cylinder with a live control panel.

Same physics and rendering as `yarn_cylinder_ogc_vispy.py`, but wrapped in a
multi-process UI:

    Parent process  —  Tkinter control panel with sliders for physics + OGC
                       parameters and Start / Pause / Reset / Stop buttons.
                       Commands are pushed onto a multiprocessing.Queue.

    Child process   —  Runs Warp, the OGCSimulation, and the vispy 3-D window.
                       Polls the queue every vispy timer tick and applies
                       parameter updates live (some — like cylinder radius —
                       also trigger an obstacle rebuild).

Running:
    python examples/yarn_cylinder_ogc_gui.py

Prerequisites:
    On Linux, tkinter ships with the system Python but may need an extra
    package on minimal installs:
        sudo apt install -y python3-tk
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys


# Compute these in the parent process so the spawned child can resolve imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Shared parameter defaults (UI starts here; worker reads them on boot) ─────

DEFAULTS = {
    "gravity_y":        -9.81,
    "particle_mass":     0.01,
    "damping":           0.996,
    "stretch_stiff":     1.0,
    "bend_stiff":        0.0,
    "substeps":          20,
    "constraint_iter":   1,
    "ogc_r":             0.005,
    "ogc_stiff":         1.0,
    "mu_static":         0.4,
    "mu_kinetic":        0.2,
    "v_max":             20.0,
    "cyl_radius":        0.05,
    "cyl_x":             1.0,
    "cyl_y":            -0.55,
    "cyl_z":             0.0,
    "yarn_x":            0.0,
    "yarn_y":            0.0,
    "yarn_z":            0.0,
}


# ── Simulation worker process ─────────────────────────────────────────────────

def sim_worker(cmd_queue, script_dir: str, defaults: dict):
    """Run Warp + OGC + vispy in a dedicated process until 'stop' is received."""
    # Imports are deferred so the CUDA / Qt context is created only in the child.
    sys.path.insert(0, os.path.join(script_dir, ".."))   # repo root → config, kernels
    sys.path.insert(0, script_dir)                        # → ogc package

    import queue as py_queue
    import numpy as np
    import warp as wp
    from vispy import app, scene
    from vispy.scene import visuals

    import config
    from ogc.mesh       import build_cylinder, mesh_for_render
    from ogc.algorithm1 import ObstacleGPU
    from ogc.algorithm3 import OGCSimulation

    # ── Scene constants ───────────────────────────────────────────────────────
    CYL_HALF_H = 1.5
    CYL_N_SEGS = 48
    BG_COL     = (0.07, 0.07, 0.13, 1.0)
    ANCHOR_COL = (0.0,  0.83, 1.0)
    FREE_COL   = (1.0,  0.65, 0.0)
    CYL_COL    = (0.25, 0.70, 0.45, 0.75)

    STRIPE_SIZE    = 5
    STRIPE_PALETTE = np.array([
        [0.95, 0.25, 0.35, 1.0],   # red
        [0.98, 0.80, 0.12, 1.0],   # gold
        [0.18, 0.82, 0.95, 1.0],   # cyan
        [0.65, 0.40, 0.95, 1.0],   # violet
    ], dtype=np.float32)

    def make_yarn_colors(n):
        idx = (np.arange(n) // STRIPE_SIZE) % len(STRIPE_PALETTE)
        return STRIPE_PALETTE[idx]

    # ── Mutable state ────────────────────────────────────────────────────────
    state   = dict(defaults)
    running = [False]           # list so nested functions can rebind
    frame   = [0]

    # ── Init Warp + sim ──────────────────────────────────────────────────────
    wp.init()
    cyl = build_cylinder(
        state["cyl_x"], state["cyl_y"], state["cyl_z"],
        state["cyl_radius"], CYL_HALF_H, n_segs=CYL_N_SEGS,
    )
    sim = OGCSimulation(
        obstacle_mesh     = cyl,
        contact_radius    = state["ogc_r"],
        contact_stiffness = state["ogc_stiff"],
    )

    def apply_state():
        """Push slider values into config + sim so the next step() uses them."""
        config.GRAVITY         = wp.vec3(0.0, float(state["gravity_y"]), 0.0)
        sim.particle_mass      = float(state["particle_mass"])
        config.DAMPING         = float(state["damping"])
        config.STRETCH_STIFF   = float(state["stretch_stiff"])
        config.BEND_STIFF      = float(state["bend_stiff"])
        config.SUBSTEPS        = int(state["substeps"])
        config.CONSTRAINT_ITER = int(state["constraint_iter"])
        sim.r                  = float(state["ogc_r"])
        sim.contact_stiffness  = float(state["ogc_stiff"])
        sim.mu_static          = float(state["mu_static"])
        sim.mu_kinetic         = float(state["mu_kinetic"])
        sim.v_max              = float(state["v_max"])

    apply_state()

    # ── Warm up Warp kernels (first step triggers JIT compile ~ a few seconds).
    # Doing this up-front means the first 'Start' click is responsive instead
    # of hanging the vispy window while kernels compile.
    print(f"[sim_worker] device={sim.device} — warming up kernels "
          f"(first-time JIT compile)...", flush=True)
    sim.step()
    sim.reset()
    print("[sim_worker] ready — GUI is live.", flush=True)

    # ── vispy scene ──────────────────────────────────────────────────────────
    canvas = scene.SceneCanvas(
        title="OGC yarn–cylinder (GUI-controlled)",
        size=(960, 720), bgcolor=BG_COL, keys="interactive", show=True,
    )
    view = canvas.central_widget.add_view()
    view.camera = scene.cameras.TurntableCamera(
        elevation=20, azimuth=-60, distance=5.5, up="y", fov=45,
    )
    visuals.XYZAxis(parent=view.scene)

    verts, faces = mesh_for_render(cyl)
    cyl_visual = visuals.Mesh(
        vertices=verts, faces=faces, color=CYL_COL, shading="smooth",
        parent=view.scene,
    )

    p = sim.positions()
    yarn_colors = make_yarn_colors(p.shape[0])
    yarn_line = visuals.Line(
        pos=p, color=yarn_colors, width=3, connect="strip", parent=view.scene,
    )
    anchor   = visuals.Markers(parent=view.scene)
    free_end = visuals.Markers(parent=view.scene)
    anchor.set_data(p[:1],   face_color=ANCHOR_COL, size=14, edge_width=0)
    free_end.set_data(p[-1:], face_color=FREE_COL,   size=14, edge_width=0)

    hud = visuals.Text(
        text="", color="white", font_size=10, anchor_x="left", anchor_y="top",
        parent=canvas.scene, pos=(10, 20),
    )

    last_cyl_r   = [state["cyl_radius"]]
    last_cyl_pos = [state["cyl_x"], state["cyl_y"], state["cyl_z"]]

    def rebuild_obstacle():
        """Replace the cylinder mesh and the GPU obstacle arrays."""
        new_mesh = build_cylinder(
            float(state["cyl_x"]), float(state["cyl_y"]), float(state["cyl_z"]),
            float(state["cyl_radius"]), CYL_HALF_H, n_segs=CYL_N_SEGS,
        )
        sim.obstacle = ObstacleGPU(new_mesh, sim.device)
        vv, ff = mesh_for_render(new_mesh)
        cyl_visual.set_data(vertices=vv, faces=ff, color=CYL_COL)
        last_cyl_r[0]   = float(state["cyl_radius"])
        last_cyl_pos[0] = float(state["cyl_x"])
        last_cyl_pos[1] = float(state["cyl_y"])
        last_cyl_pos[2] = float(state["cyl_z"])

    # ── Main tick: drain queue, step sim, update visuals ─────────────────────
    def on_timer(_event):
        try:
            while True:
                cmd = cmd_queue.get_nowait()
                kind = cmd[0]
                print(f"[sim_worker] cmd: {cmd}", flush=True)
                if   kind == "start": running[0] = True
                elif kind == "pause": running[0] = False
                elif kind == "reset":
                    running[0] = False
                    sim.reset()
                    frame[0] = 0
                elif kind == "stop":
                    app.quit()
                    return
                elif kind == "param":
                    key, value = cmd[1], cmd[2]
                    state[key] = value
                    if key == "cyl_radius" and abs(float(value) - last_cyl_r[0]) > 1e-4:
                        rebuild_obstacle()
                    elif key in ("cyl_x", "cyl_y", "cyl_z"):
                        rebuild_obstacle()
                    elif key in ("yarn_x", "yarn_y", "yarn_z"):
                        sim.yarn_origin = np.array(
                            [state["yarn_x"], state["yarn_y"], state["yarn_z"]],
                            dtype=np.float32,
                        )
                        sim.reset()
                        frame[0] = 0
                    apply_state()
        except py_queue.Empty:
            pass

        if running[0]:
            sim.step()
            frame[0] += 1

        pp = sim.positions()
        yarn_line.set_data(pos=pp)
        anchor.set_data(pp[:1],   face_color=ANCHOR_COL, size=14, edge_width=0)
        free_end.set_data(pp[-1:], face_color=FREE_COL,   size=14, edge_width=0)
        status = "RUN" if running[0] else "PAUSED"
        hud.text = (
            f"[{sim.device}]  frame {frame[0]:05d}  {status}  t={sim.time:.2f}s\n"
            f"r={sim.r:.3f}  k_c={sim.contact_stiffness:.2f}  "
            f"substeps={config.SUBSTEPS}  iter={config.CONSTRAINT_ITER}"
        )
        canvas.update()

    # IMPORTANT: keep a reference to the Timer.  vispy's Timer is not
    # otherwise held by the backend in some configurations, so without this
    # local binding it can be garbage-collected and stop firing — which makes
    # the GUI look frozen and ignore Start clicks.
    timer = app.Timer(interval=1.0 / 60.0, connect=on_timer, start=True)
    print("[sim_worker] entering vispy event loop", flush=True)
    app.run()
    del timer


# ── Tkinter control panel (parent process) ───────────────────────────────────

def run_ui(cmd_queue):
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("OGC yarn–cylinder — controls")
    root.geometry("460x870")

    def add_slider(label, key, from_, to_, default, is_int=False, fmt="{:.3f}"):
        frm = ttk.Frame(root)
        frm.pack(fill="x", padx=8, pady=4)
        ttk.Label(frm, text=label, width=20).pack(side="left")

        val_var  = tk.DoubleVar(value=default)
        # Int sliders show  "current / max"  so the ceiling is always visible.
        if is_int:
            init_disp = f"{int(default)} / {int(to_)}"
        else:
            init_disp = fmt.format(default)
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

    ttk.Label(root, text="Physics", font=("", 10, "bold")).pack(anchor="w", padx=8, pady=(10, 0))
    add_slider("Gravity Y",        "gravity_y",      -20.0, 0.0,  DEFAULTS["gravity_y"],   fmt="{:+.2f}")
    add_slider("Particle mass (kg)","particle_mass",   0.01, 10.0, DEFAULTS["particle_mass"], fmt="{:.3f}")
    add_slider("Damping",          "damping",         0.90, 1.00, DEFAULTS["damping"])
    add_slider("Stretch stiff",    "stretch_stiff",   0.0,  1.0,  DEFAULTS["stretch_stiff"])
    add_slider("Bend stiff",       "bend_stiff",      0.0,  1.0,  DEFAULTS["bend_stiff"])
    add_slider("Substeps",         "substeps",        1,   200,   DEFAULTS["substeps"],        is_int=True)
    add_slider("Constraint iter",  "constraint_iter", 1,    50,   DEFAULTS["constraint_iter"], is_int=True)

    ttk.Label(root, text="OGC contact", font=("", 10, "bold")).pack(anchor="w", padx=8, pady=(10, 0))
    add_slider("Contact radius r (m)", "ogc_r",        0.001, 0.20, DEFAULTS["ogc_r"], fmt="{:.4f}")
    add_slider("Contact stiffness","ogc_stiff",       0.0,   1.0,  DEFAULTS["ogc_stiff"])
    add_slider("Friction μ_static", "mu_static",      0.0,   1.0,  DEFAULTS["mu_static"])
    add_slider("Friction μ_kinetic","mu_kinetic",      0.0,   1.0,  DEFAULTS["mu_kinetic"])
    add_slider("Velocity max (m/s)","v_max",           0.0,  100.0, DEFAULTS["v_max"],    fmt="{:.1f}")

    ttk.Label(root, text="Obstacle", font=("", 10, "bold")).pack(anchor="w", padx=8, pady=(10, 0))
    add_slider("Cylinder radius (m)","cyl_radius",     0.02,  0.50, DEFAULTS["cyl_radius"])
    add_slider("Cylinder X",         "cyl_x",         -3.0,   3.0,  DEFAULTS["cyl_x"],  fmt="{:+.3f}")
    add_slider("Cylinder Y",         "cyl_y",         -3.0,   3.0,  DEFAULTS["cyl_y"],  fmt="{:+.3f}")
    add_slider("Cylinder Z",         "cyl_z",         -3.0,   3.0,  DEFAULTS["cyl_z"],  fmt="{:+.3f}")

    ttk.Label(root, text="Yarn origin", font=("", 10, "bold")).pack(anchor="w", padx=8, pady=(10, 0))
    add_slider("Origin X",           "yarn_x",        -3.0,   3.0,  DEFAULTS["yarn_x"], fmt="{:+.3f}")
    add_slider("Origin Y",           "yarn_y",        -3.0,   3.0,  DEFAULTS["yarn_y"], fmt="{:+.3f}")
    add_slider("Origin Z",           "yarn_z",        -3.0,   3.0,  DEFAULTS["yarn_z"], fmt="{:+.3f}")

    # ── Buttons ──────────────────────────────────────────────────────────────
    def send(cmd: str):
        print(f"[ui] send: {cmd}", flush=True)
        cmd_queue.put((cmd,))

    btn_frm = ttk.Frame(root)
    btn_frm.pack(fill="x", padx=8, pady=16)
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
    # Use spawn so the child process starts fresh — avoids inheriting a Warp /
    # CUDA context from the parent, and works identically on Linux / macOS / Win.
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
