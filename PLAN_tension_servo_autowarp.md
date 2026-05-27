# Plan: Tension Servo + Auto-warp Initial State

Target file: `examples/yarn_rolls_ogc_gui.py` (and related OGC helpers as needed)

---

## Phase 1 — Auto-warp initial yarn geometry

**Goal:** `make_initial_positions()` produces a physically correct starting state where the
yarn already wraps the guide cylinder by a specified angle.

### What changes

- Add a new `initial_warp_angle` parameter (degrees, default `90`) with a slider in the GUI.
- Add a geometry helper that, given Roll A position/radius, guide cylinder position/radius,
  and Roll B position/radius, computes:
  1. The **external tangent line** from Roll A surface to the guide cylinder — giving the
     tangent-in point on the guide.
  2. The **arc** on the guide surface (`r_cyl + ogc_r`) spanning `initial_warp_angle` degrees
     from tangent-in, in the direction determined by the Roll A → guide → Roll B geometry.
  3. The **external tangent line** from the guide cylinder out to Roll B — giving the
     tangent-out point.
- Distribute the `N` particles along the full path:
  wound coil on Roll A → straight free span to tangent-in → arc on guide → straight free
  span to Roll B, with spacing proportional to `REST_LEN`.
- The wound section on Roll A is unchanged.

### Files touched

- `make_initial_positions()` — replace straight free-span logic with warped geometry
- `DEFAULTS` — add `initial_warp_angle`
- GUI slider section — add the new slider
- `sim_reset()` — no change needed (already calls `make_initial_positions`)

---

## Phase 2 — Auto-position sensor A and sensor B

**Goal:** Sensors A and B always sit at the midpoint of their respective yarn spans, oriented
perpendicular to the yarn direction, so they reliably capture tension without manual placement.

### What changes

- Add helper `_auto_place_sensors()` that:
  - Computes tangent-in and tangent-out points (same geometry as Phase 1).
  - Places **sensor A** at the midpoint of Roll A departure → guide tangent-in; sets
    `sensor_a_hx/hy/hz` so the detection box is a thin plate perpendicular to the yarn.
  - Places **sensor B** at the midpoint of guide tangent-out → Roll B attachment; same logic.
  - Updates `state[...]` and sends `("param", ...)` commands to sync the simulation worker.
- Call `_auto_place_sensors()` at startup and whenever `rebuild_roll_a()`, `rebuild_roll_b()`,
  or `rebuild_guide()` is triggered.
- Manual slider control still works and overrides auto-placement until the next rebuild.

### Files touched

- `_auto_place_sensors()` — new helper function
- `rebuild_roll_a()`, `rebuild_roll_b()`, `rebuild_guide()` — call auto-place after rebuild
- `do_reinit()` — call auto-place after reinit

### Dependency

Requires Phase 1 (needs the tangent-point geometry computation).

---

## Phase 3 — Roll A tension servo

**Goal:** Roll A actively controls upstream tension (T_a) to a setpoint by acting as a full
servo — both braking (resist yarn paying out too fast) and driving (feeding yarn to prevent
over-tension). An on/off switch keeps the original passive flywheel behavior available.

### What changes

**New parameters**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `roll_a_servo_on` | checkbox | `0` (off) | Enable/disable servo control |
| `roll_a_tension_setpoint` | slider (0–50 cN) | `5.0` | Target upstream tension |
| `roll_a_kp` | slider | `1.0` | Proportional gain |
| `roll_a_ki` | slider | `0.1` | Integral gain |

**Control loop** (Python side, `on_timer`, every rendered frame)

- If servo is **off**: passive flywheel behavior unchanged.
- If servo is **on**:
  - Read `T_a = shared[0]`.
  - Compute `error = setpoint - T_a`.
  - Accumulate integral term.
  - Compute `omega_command = kp * error + ki * integral`.
  - Send `("param", "roll_a_omega_override", omega_command)` to simulation worker.

**Simulation worker** (`_execute_substeps`)

- If servo is **on**: set `omega_a_wp` to `omega_command` before `roll_a_torque_step`,
  bypassing the yarn-driven flywheel dynamics.
- If servo is **off**: unchanged — `roll_a_torque_step` drives omega from yarn torque.

**HUD addition**

```
SERVO: ON  setpoint=X.Xcn  err=±Y.YcN
```

### Files touched

- `_execute_substeps()` — servo branch before `roll_a_torque_step`
- `on_timer()` — PI control loop
- `DEFAULTS` — add servo parameters
- GUI slider/checkbox section — add new controls
- `_snapshot_params()` — include `roll_a_servo_on` so CUDA graph rebuilds on toggle

### Dependency

Independent of Phases 1 and 2, but reliable tension readings from correctly placed sensors
(Phase 2) are needed for meaningful servo tuning.

---

## Implementation order

```
Phase 1  →  Phase 2  →  Phase 3
```

Phase 1 and 2 are tightly coupled (share tangent-point geometry).
Phase 3 can be started in parallel but benefits from Phase 2 sensor placement.
