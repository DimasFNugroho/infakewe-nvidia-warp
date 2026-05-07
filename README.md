# Yarn Simulation — NVIDIA Warp PBD

A real-time 3-D yarn simulation using **Position-Based Dynamics** (PBD) running
on the GPU via [NVIDIA Warp](https://github.com/NVIDIA/warp). The yarn is a
chain of particles connected by stretch + bending constraints. Several demos
are provided (wind sway, free-fall, collisions with a cylinder or tilted
plane, with or without friction).

---

## 1. Requirements

| Component        | Notes                                                    |
| ---------------- | -------------------------------------------------------- |
| Python           | 3.9 or newer                                             |
| OS               | Linux, Windows, or macOS                                 |
| GPU (optional)   | NVIDIA CUDA GPU for GPU acceleration (falls back to CPU) |

### Python packages

The core simulation needs just **warp-lang** and **numpy**. Each example
picks one of three visualisation backends — install only the ones you want.

| Purpose                                | Package         |
| -------------------------------------- | --------------- |
| Physics engine (always required)       | `warp-lang`     |
| Numerics (always required)             | `numpy`         |
| Default viewer + matplotlib examples   | `matplotlib`    |
| Vispy examples (`*_vispy.py`)          | `vispy`, `PyQt5`|
| Polyscope examples (`yarn_freefall.py`, `yarn_wind_sway.py`) | `polyscope` |

### System libraries (Linux / WSL2 only)

PyQt5 needs some X11 / xcb system libraries that aren't always present on
minimal Linux installs (including WSL2). A helper script is provided —
run it once:

```bash
./install_system_deps.sh
```

The script uses `apt-get` (Debian / Ubuntu / WSL2) and asks for `sudo`
if needed. It installs: `libxcb-xinerama0`, `libxcb-cursor0`,
`libxcb-icccm4`, `libxcb-image0`, `libxcb-keysyms1`, `libxcb-randr0`,
`libxcb-render-util0`, `libxcb-shape0`, `libxcb-sync1`, `libxcb-xfixes0`,
`libxcb-xkb1`, `libxkbcommon-x11-0`, `libxkbcommon0`, `libdbus-1-3`.

On WSL2 with WSLg you can also skip xcb entirely by using the Wayland
backend:

```bash
QT_QPA_PLATFORM=wayland python examples/yarn_freefall_vispy.py
```

---

## 2. Installation

```bash
# 1. Clone
git clone <this-repo-url>
cd infakewe-nvidia-warp

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Upgrade pip and install all dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

To leave the virtual environment later, run `deactivate`. Re-activate it
with the same `source .venv/bin/activate` command before running the
scripts again.

Verify the install:

```bash
python -c "import warp as wp; wp.init(); print('Warp', wp.__version__, 'CUDA:', wp.is_cuda_available())"
```

---

## 3. Running the default simulation

From the repo root:

```bash
python main.py                 # auto: GPU if available, else CPU
python main.py --cpu           # force CPU
python main.py --fps 30        # lower the animation frame rate
```

This opens a matplotlib window showing a yarn anchored at the top, swinging
under gravity + oscillating wind. Close the window to exit.

---

## 4. Running the examples

All example scripts are self-contained and must be run **from the repo root**
(they add the parent directory to `sys.path` so `config`, `kernels`,
`simulation` resolve correctly):

```bash
# Polyscope viewers
python examples/yarn_freefall.py
python examples/yarn_wind_sway.py

# Matplotlib viewer (no GPU viewer deps)
python examples/yarn_freefall_matplotlib.py

# Vispy viewers (GPU-accelerated OpenGL)
python examples/yarn_freefall_vispy.py
python examples/yarn_cylinder_vispy.py
python examples/yarn_cylinder_friction_vispy.py
python examples/yarn_plane_friction_vispy.py

# OGC (Offset Geometric Contact, SIGGRAPH 2025) demo
python examples/yarn_cylinder_ogc_vispy.py

# OGC demo with a live Tkinter control panel (runs the sim in a subprocess)
python examples/yarn_cylinder_ogc_gui.py
```

The `yarn_cylinder_ogc_vispy.py` example replaces the analytic cylinder
collision with a full OGC contact pipeline implemented in
`examples/ogc/algorithm{1,2,3,4}.py` (Warp kernels). The cylinder is
treated as a triangulated mesh plus an offset radius `r`.

The `yarn_cylinder_ogc_gui.py` variant spawns the simulation + vispy
viewer in a child process and drives it from a Tkinter control panel in
the parent: sliders for gravity, damping, stretch/bend stiffness,
substeps, OGC radius, contact stiffness, and cylinder radius, plus
Start / Pause / Reset / Exit buttons. On Linux you may need
`sudo apt install -y python3-tk` for tkinter.

### Vispy controls

| Key / Mouse    | Action           |
| -------------- | ---------------- |
| Space          | Pause / resume   |
| R              | Reset simulation |
| Left-drag      | Orbit camera     |
| Middle-drag    | Pan              |
| Scroll         | Zoom             |

### Polyscope controls

| Input        | Action        |
| ------------ | ------------- |
| Left-drag    | Rotate camera |
| Right-drag   | Pan           |
| Scroll       | Zoom          |
| Space        | Pause         |

---

## 5. Tweaking the simulation

Global defaults live in `config.py`. The most useful knobs:

| Parameter       | Meaning                                              |
| --------------- | ---------------------------------------------------- |
| `NUM_PARTICLES` | Number of particles in the yarn                      |
| `YARN_LENGTH`   | Total rest length, in metres                         |
| `DT`            | Physics timestep (default 1/60 s)                    |
| `SUBSTEPS`      | Sub-steps per frame — higher = more stable, slower   |
| `BEND_STIFF`    | 0.0 floppy noodle → 0.9 stiff wire                   |
| `DAMPING`       | Per-substep velocity damping (closer to 1.0 = less)  |
| `WIND_AMP_X/Z`  | Wind amplitude along each axis                       |

Friction-based examples expose their own `MU_STATIC` / `MU_KINETIC` constants
at the top of each script.

---

## 6. Project layout

```
.
├── main.py              # entry point — matplotlib viewer
├── config.py            # simulation parameters
├── simulation.py        # Simulation class (state + step())
├── kernels.py           # Warp GPU kernels (PBD integrate / stretch / bend)
├── visualization.py     # matplotlib 3-D visualiser
└── examples/            # standalone demo scripts
    ├── yarn_freefall.py                    # polyscope
    ├── yarn_freefall_matplotlib.py         # matplotlib
    ├── yarn_freefall_vispy.py              # vispy
    ├── yarn_wind_sway.py                   # polyscope
    ├── yarn_cylinder_vispy.py              # yarn falling onto cylinder
    ├── yarn_cylinder_friction_vispy.py     # + Coulomb friction
    ├── yarn_plane_friction_vispy.py        # yarn on tilted plane + friction
    ├── yarn_cylinder_ogc_vispy.py          # cylinder collision via OGC pipeline
    ├── yarn_cylinder_ogc_gui.py            # OGC demo + Tkinter control panel (multiproc)
    └── ogc/                                # OGC (SIGGRAPH 2025) modules
        ├── mesh.py                         # triangulated-mesh + feature normals
        ├── algorithm1.py                   # Vertex-Facet contact detection (Warp)
        ├── algorithm2.py                   # Edge-Edge  contact detection (Warp)
        ├── algorithm3.py                   # Simulation step orchestrator
        └── algorithm4.py                   # Inner iteration: stretch+bend+OGC project
```

---

## 7. Tension sensing and the Capstan equation

The roll-to-roll simulation (`yarn_rolls_ogc_gui.py`) includes two virtual
tension sensors placed on either side of the guide cylinder. The upstream
sensor measures **T_A** (hold-side tension) and the downstream sensor
measures **T_B** (load-side tension).

### How tension is computed

Tension is derived from the PBD position corrections. For a segment between
particles `i` and `i+1` with rest length `L₀`:

```
extension  = |p_{i+1} − p_i| − L₀
T  (N)     = particle_mass × stretch_stiff × extension / sub_dt²
T  (cN)    = T × 100
```

This follows from the PBD impulse equivalence: the per-substep position
correction `Δx = stiff × extension` is equivalent to a spring impulse
`F·Δt = m·Δx`, giving `F = m·stiff·extension / Δt²`. The result is
averaged over a small window of segments around the sensor particle to
suppress noise.

### Capstan equation validation

The Capstan equation for a yarn sliding over a cylinder under kinetic
friction is:

```
T_B / T_A = e^(μ_k · θ)
```

where `θ` is the **wrap angle** — the arc subtended by the yarn contact on
the guide cylinder (in radians). The simulation computes `θ` every frame
from the angle between the vectors from the guide centre to each sensor
particle.

The panel displays:
- **Measured ratio** `T_B / T_A` — from particle positions
- **Theoretical ratio** `e^(μ_k · θ)` — Capstan prediction using the
  simulation's own `μ_k` and the measured `θ`
- **Residual** `measured / theoretical` — how well the friction model
  matches the Capstan equation; 1.0 = perfect agreement

### Important limitation

All three quantities (`T_A`, `T_B`, `θ`) are derived from the same PBD
simulation state. The residual therefore measures **internal consistency**
— whether the tension ratio the simulation produces matches the Capstan
amplification factor for the geometry it produces. It is *not* a
ground-truth validation against physical measurements.

Additionally, the absolute cN values depend on the `particle_mass`,
`stretch_stiff`, and `substeps` settings. If any of these change, the
scale of the readings changes accordingly. The readings are most meaningful
when compared *relative to each other* (i.e., the ratio and residual)
rather than as absolute forces.

---

## 8. Troubleshooting

- **`CUDA available: False`** — Warp falls back to CPU automatically. If you
  expected GPU, check that an NVIDIA driver is installed and that
  `nvidia-smi` works.
- **`ModuleNotFoundError: config` / `simulation`** — you ran an example
  script from inside `examples/`. Run it from the repo root instead.
- **Vispy window is black or won't open** — ensure a Qt backend is
  installed (`pip install PyQt5`). On Linux, an X / Wayland session is
  required; over SSH use `ssh -X` or switch to the matplotlib example.
- **`Could not load the Qt platform plugin "xcb"`** (Linux / WSL2) — run
  `./install_system_deps.sh`, or launch with `QT_QPA_PLATFORM=wayland`. To
  identify the specific missing library, run once with `QT_DEBUG_PLUGINS=1`
  and look for the first
  `Cannot load library ... cannot open shared object file` line.
- **Polyscope import error** — `pip install polyscope`. Requires OpenGL 3.3+.
