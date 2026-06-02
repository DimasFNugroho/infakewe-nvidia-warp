# Plan: Collocated Anchor-Tension Feedback for Roll A Servo

Target file: `examples/yarn_rolls_ogc_gui.py`

Motivation: the current PI servo reads `T_A` from Sensor A at the midpoint of
the Roll A → guide free span. Elastic yarn between the actuator (Roll A
surface) and the sensor introduces a transport delay — the controller
reacts to history rather than the present, and at moderate gains this
turns into ringing/instability. Moving the feedback to the segment that
*starts at* the kinematic anchor on Roll A gives a collocated sensor /
actuator pair, which is the classical control-theory recipe for stable
feedback on a flexible system.

---

## Phase A — Anchor-tension measurement

**Goal:** compute a per-frame upstream tension reading from the first
`k` yarn segments starting at the kinematic anchor `particle 0`, and
expose it as the servo's process variable instead of `shared[0]`.

### Formula

For each segment `i = 0, 1, …, k−1`:
```
ext_i = max(0, ‖x_{i+1} − x_i‖ − L_0)
T_i   = m_p · k_s · ext_i / (Δt_ref)²            (Newtons)
```
where `Δt_ref = config.DT / 200` is the reading-normalisation constant
already used by the existing sensor (so absolute cN values remain
comparable to today's Sensor A reading).

The frame's raw anchor tension is the mean:
```
T_anchor_raw = (100 N→cN) · mean(T_0, …, T_{k−1})
```

### New parameters

| Key                        | Type     | Default | Description |
|----------------------------|----------|---------|-------------|
| `roll_a_servo_source`      | int      | 1 (anchor) | 0 = Sensor A, 1 = anchor segments |
| `roll_a_anchor_k`          | int      | 3       | Number of segments averaged (1…20) |
| `roll_a_tension_ewma`      | float    | 0.20    | EWMA smoothing factor α ∈ [0,1] (1 = no smoothing) |

### EWMA filter

State variable `_T_anchor_filt[0]` initialised to 0. Each frame:
```
T_anchor_filt = α · T_anchor_raw + (1 − α) · T_anchor_filt
```
Reset to 0 on `sim_reset()` and when the servo is toggled off.

### Files / functions touched

- `DEFAULTS` — add the three keys above.
- Module-level mutable state — `_T_anchor_filt = [0.0]` next to
  `_servo_integral`.
- `on_timer` — compute `T_anchor_raw` and `T_anchor_filt` every frame
  (whether or not the servo is on, so the HUD can show it for
  comparison).
- PI loop — pick the process variable from `roll_a_servo_source`:
  `T_proc = T_anchor_filt if source == 1 else shared[0]`.
- `sim_reset()` — zero `_T_anchor_filt`.

---

## Phase B — HUD readback

Keep the Sensor A reading visible so the user can *see* the propagation
delay. New HUD line under the existing tension line:
```
T_anchor=X.XXcN  (filt)  src=anchor|sensor  k=K  α=0.20
```

---

## Phase C — Tkinter GUI

Add a new section **"Roll A servo source"** above the existing
"Roll A — tension servo (PI on T_A)" section:

- A dropdown (`ttk.Combobox` or radio buttons) for `roll_a_servo_source`:
  *"Sensor A (free-span midpoint)"* vs *"Anchor segments (collocated)"*.
- An integer slider for `roll_a_anchor_k`, range 1–20, default 3.
- A float slider for `roll_a_tension_ewma`, range 0.0–1.0, default 0.20.

The existing servo section is left untouched; setpoint / kp / ki keep
their semantics, only the meaning of the *measured* signal changes.

---

## Validation

Two scenarios to confirm the fix:

1. **Step setpoint test.** Servo on, run until steady, then move
   `roll_a_tension_setpoint` from 5 cN → 10 cN. Compare overshoot and
   settling time between `source=sensor` and `source=anchor` with
   identical `kp/ki`. Anchor should settle faster with less ringing.

2. **High-gain stability boundary.** Increase `kp` until the loop just
   becomes unstable. The anchor-feedback case should tolerate
   substantially higher gain before going unstable. Document the
   approximate gain margin difference.

---

## Out of scope (to discuss later)

- Yarn feeder behaviour reform — separate discussion thread; this plan
  only addresses where the controller takes its measurement, not how
  the feeder responds to the command.
