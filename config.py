"""config.py — Yarn simulation parameters.

Tweak these values to change the yarn's behaviour:
  - BEND_STIFF  : 0.0 = floppy noodle, 0.9 = stiff wire
  - DAMPING     : closer to 1.0 = less air resistance
  - SUBSTEPS    : higher = more stable physics, slower per frame
  - WIND_AMP_*  : how hard the wind blows
"""

import warp as wp

# ── Yarn geometry ─────────────────────────────────────────────────────────────
NUM_PARTICLES = 50
YARN_LENGTH   = 2.0                              # total rest length (m)
REST_LEN      = YARN_LENGTH / (NUM_PARTICLES - 1)

# ── Time integration ──────────────────────────────────────────────────────────
DT             = 1.0 / 60.0   # physics timestep (s)
SUBSTEPS       = 20            # sub-steps per frame
CONSTRAINT_ITER = 10           # constraint solver iterations per sub-step

# ── Physics ───────────────────────────────────────────────────────────────────
GRAVITY       = wp.vec3(0.0, -9.81, 0.0)
DAMPING       = 0.998          # velocity damping per sub-step  [0..1]
STRETCH_STIFF = 1.0            # stretch constraint stiffness   [0..1]
BEND_STIFF    = 0.25           # bending constraint stiffness   [0..1]

# ── Wind ──────────────────────────────────────────────────────────────────────
WIND_FREQ  = 0.35   # oscillation frequency (Hz)
WIND_AMP_X = 3.0    # amplitude along X axis
WIND_AMP_Z = 2.0    # amplitude along Z axis
