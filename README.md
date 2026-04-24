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
```

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
    └── yarn_plane_friction_vispy.py        # yarn on tilted plane + friction
```

---

## 7. Troubleshooting

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
