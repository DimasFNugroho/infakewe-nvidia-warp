# Plan: Roll B Torque-Limited Motor

Branch: `roll-b/torque-limit`
Predecessor: `master` (no dependency on the feeder branches; can land
in parallel).

This plan addresses the downstream asymmetry called out during the
feeder discussion: Roll B is currently a *pure kinematic motor* — it
advances `angle_b` at the prescribed `pull_speed` regardless of yarn
state. If the feeder is jammed or the yarn elsewhere is stuck, Roll B
will happily stretch segment $[\,N-2,\,N-1\,]$ to infinity. We replace
the kinematic motor with a **velocity-controlled torque-limited
flywheel**, mirroring the existing passive-flywheel logic on Roll A
but driving the yarn instead of being driven by it.

---

## 1. Goal

Roll B behaves as a real DC-motor-driven pulling roll:

- It has a programmed target speed $v_\text{pull}$ (already a slider).
- A velocity controller produces a drive torque $\tau_\text{drive}$
  proportional to the speed error.
- $\tau_\text{drive}$ is clamped to $\pm\tau_\text{max}$ — the motor
  cannot deliver more torque than its hardware limit.
- The yarn pulls back on Roll B through the kinematic last particle;
  that load torque $\tau_\text{yarn}$ is read from the anchor-side
  segment $[\,N-2,\,N-1\,]$.
- Roll B integrates $\dot\omega_B = (\tau_\text{drive} - \tau_\text{yarn})/I_B$
  per substep.

Consequences:

- If yarn tension is low → $\tau_\text{yarn}$ small → $\omega_B$
  reaches and holds $v_\text{pull}/r_B$. Backwards-compatible.
- If yarn tension is high (feeder stalled, downstream stuck) →
  $\tau_\text{yarn} > \tau_\text{max}$ → motor cannot match it →
  $\omega_B$ decelerates → can fall to zero → Roll B *stalls*. The
  excess load is then carried as static yarn tension. The simulator
  stays physical.

---

## 2. New parameters

| Key                   | Type  | Default | Range    | Description |
|-----------------------|-------|---------|----------|-------------|
| `roll_b_torque_limit_on` | int   | 1       | 0/1      | Master switch. 0 = legacy kinematic behaviour. |
| `roll_b_max_torque`   | float | 5.0     | 0.01–100 | $\tau_\text{max}$ (N·m). High default keeps limit invisible at light loads. |
| `roll_b_mass`         | float | 0.5     | 0.01–5   | Roll B mass (kg) — sets inertia $I_B = \tfrac{1}{2} M_B r_B^2$. |
| `roll_b_bearing_damping` | float | 0.998  | 0–1      | Per-substep multiplicative damping on $\omega_B$ (mirrors Roll A). |
| `roll_b_drive_gain`   | float | 50.0    | 1–500    | $K_v$ — drive torque per unit of speed error (N·m·s/rad). |

`roll_b_torque_limit_on = 0` falls back to the existing
`roll_b_motor_step` kernel verbatim, so all current behaviour is
preserved as a slider toggle.

---

## 3. Conceptual model

Let $\omega_B^*$ be the target angular speed, $\omega_B$ the current
one, $T_B^\text{anchor}$ the tension in the anchor segment, $r_B$ the
physical radius, $I_B = \tfrac{1}{2}M_B r_B^2$ the inertia.

Each substep:

$$
\begin{aligned}
\omega_B^*       &= v_\text{pull} / r_B \\
\tau_\text{drive} &= \mathrm{clamp}\big(K_v\,(\omega_B^* - \omega_B),\; -\tau_\text{max},\; \tau_\text{max}\big) \\
\tau_\text{yarn} &= T_B^\text{anchor} \cdot r_B \cdot \tau_\text{sign} \\
\Delta\omega_B    &= \frac{\tau_\text{drive} - \tau_\text{yarn}}{I_B}\,\Delta t' \\
\omega_B          &\leftarrow \alpha_\text{bear} \cdot (\omega_B + \Delta\omega_B) \\
\omega_B          &\leftarrow \mathrm{clamp}(\omega_B, -\omega_\text{max}, \omega_\text{max}) \\
\theta_B          &\leftarrow \theta_B + \omega_B \cdot \Delta t' \\
\vect{x}_{N-1}    &\leftarrow \vect{c}_B + R_B^\text{orb}\,(\cos\theta_B, \sin\theta_B, 0)
\end{aligned}
$$

$\tau_\text{sign} = +1$ if the anchor segment pulls Roll B in the
direction opposite to its rotation (the normal driving case). In code
this is computed analogously to the Roll A torque kernel:
the projection of the segment direction onto the surface tangent
$\unit{t} = (-\sin\theta_B, \cos\theta_B, 0)$, with the sign chosen so
that yarn pull always *opposes* motor drive.

$T_B^\text{anchor}$ uses the same PBD impulse equivalence used in the
existing sensor:
$T = m_p \cdot k_s \cdot \max(\norm{\vect{x}_{N-1} - \vect{x}_{N-2}} - L_0, 0) / (\Delta t'_\text{ref})^2$.

---

## 4. Kernel design

A new kernel `kernel_roll_b_torque_limited_step` in
`examples/ogc/algorithm4.py` takes:

```
pos               wp.array(vec3)
center            wp.vec3
rb                float                # physical radius
orbit_r           float
rest_len          float
stretch_stiff     float
particle_mass     float
roll_mass         float                # M_B
sub_dt            float
bearing_damping   float
drive_gain        float                # K_v
max_torque        float                # tau_max
omega_max         float
target_omega      float                # v_pull / r_b, computed Python-side
n_last            int
angle             wp.array(float)      # size-1, in/out
omega             wp.array(float)      # size-1, in/out
```

Single-thread (`dim=1`) — mirrors the existing
`kernel_roll_a_torque_update` exactly in structure. The Python wrapper
is `roll_b_torque_limited_step(...)`.

The existing `kernel_roll_b_motor_step` is kept untouched as the
fallback path when `roll_b_torque_limit_on = 0`.

---

## 5. Wiring in `_execute_substeps`

Inside the substep loop, replace:

```python
roll_b_motor_step(
    pos_wp, center_b, rb, orbit_r_b,
    float(state["pull_speed"]), sub_dt,
    N - 1, angle_b_wp, device,
)
```

with a Python-side branch baked into the CUDA graph at capture:

```python
if torque_limited_b:
    roll_b_torque_limited_step(
        pos_wp, center_b, rb, orbit_r_b,
        config.REST_LEN, config.STRETCH_STIFF,
        particle_mass, M_b, sub_dt,
        float(state["roll_b_bearing_damping"]),
        float(state["roll_b_drive_gain"]),
        float(state["roll_b_max_torque"]),
        omega_max, target_omega_b,
        N - 1, angle_b_wp, omega_b_wp, device,
    )
else:
    roll_b_motor_step(...)   # existing call, unchanged
```

The `torque_limited_b` boolean is included in `_snapshot_params`, so
toggling the switch rebuilds the graph cleanly.

New module-level state next to the existing `omega_a_wp`:

```python
omega_b_wp = wp.array([0.0], dtype=float, device=device)   # new
```

and reset to zero in `sim_reset()` and `do_reinit()`.

---

## 6. Initial condition

Start $\omega_B = 0$. The motor will accelerate up to $v_\text{pull}/r_B$
in approximately $I_B / K_v$ seconds at light load — about 0.5 s with
default values. That start-up transient is physically correct and
should be visible as a brief tension drop as the simulation begins.

---

## 7. Files touched

- `examples/ogc/algorithm4.py`
  - Add `kernel_roll_b_torque_limited_step` and
    `roll_b_torque_limited_step`.
- `examples/yarn_rolls_ogc_gui.py`
  - `DEFAULTS` — add the five keys in §2.
  - Module-level — declare `omega_b_wp`. Reset on
    `sim_reset()` / `do_reinit()`.
  - `_execute_substeps` — Python-side branch as in §5.
  - `_snapshot_params` — include `roll_b_torque_limit_on`.
  - GUI — new section "Roll B — torque limit" with checkbox + four
    sliders.
  - HUD — append $\omega_B$ readout (and optionally $\tau_\text{drive}$
    saturation flag).

---

## 8. Out of scope

- A tension-feedback PI on $T_B$ for Roll B. The user explicitly
  wants speed control with a physical torque ceiling, not active
  tension regulation on the downstream side.
- A DC-motor droop curve $\tau_\text{drive}(\omega) =
  \tau_\text{stall}(1 - \omega/\omega_\text{nl})$. We use the linear
  velocity-error feedback (B2 from the discussion), which is
  approximately equivalent at the working point.
- Coupling Roll B to a knitting-machine load model. Roll B is the
  yarn-side actuator; what happens beyond it is implicit.

---

## 9. Risks

- **Sign of $\tau_\text{yarn}$.** The Roll A passive kernel projects
  the segment direction onto the surface tangent and takes whatever
  sign falls out, because both signs are physically meaningful for a
  passive roll. For Roll B, the convention must always have yarn pull
  *oppose* motor drive (otherwise the yarn would *speed up* the
  motor, which is unphysical for a driven roll). We enforce this by
  taking the absolute value of the tangential projection: yarn pull
  contributes $|\tau_\text{yarn}|$ to the load torque regardless of
  sign.
- **Bearing-damping interpretation.** On Roll A bearing damping
  represents friction in the bearing. On a motor-driven Roll B it
  represents the same thing, but together with the velocity-error
  drive term it forms a second pole in the speed response. At
  $\alpha_\text{bear} = 0.998$ this is negligible. If users dial it
  down, the speed loop may overshoot.
- **CUDA graph and `target_omega`.** $\omega_B^* = v_\text{pull}/r_B$
  is a scalar baked into the graph at capture; changing `pull_speed`
  rebuilds the graph (it is already in the snapshot). No further
  changes needed.
- **Anchor-segment tension as load proxy.** Real Roll B sees yarn
  tension distributed over its wrap, not a single segment. Using the
  $[\,N-2,\,N-1\,]$ segment as a load proxy is the same approximation
  the passive flywheel uses on Roll A — good enough for control,
  not exact for force balance.

---

## 10. Acceptance tests

1. **Backwards compatibility.** With `roll_b_torque_limit_on = 0`,
   simulation output is bit-identical to today's run. (Compare $T_A$,
   $T_B$, $\theta$ traces.)

2. **Light-load behaviour.** With `roll_b_torque_limit_on = 1` and
   defaults, downstream tension is low; $\omega_B$ reaches
   $v_\text{pull}/r_B$ within a fraction of a second and stays there.
   Indistinguishable from the kinematic baseline.

3. **Stall under blocked feeder.** Set
   `roll_a_servo_on = 1`, `roll_a_tension_setpoint = 0`, $k_p, k_i = 0$
   (Roll A effectively immobile). With torque limit on,
   $\omega_B \to 0$ within seconds; yarn does not stretch unbounded.
   With torque limit off, segment $[\,N-2,\,N-1\,]$ stretches without
   limit until the simulation explodes.

4. **Recovery.** Re-enable Roll A → tension drops → Roll B
   re-accelerates and resumes normal speed.

5. **Torque-limit sensitivity sweep.** Hold all else constant, sweep
   `roll_b_max_torque` from 0.1 → 100 N·m. Plot steady-state
   $\omega_B$ versus $\tau_\text{max}$. Below a critical
   $\tau_\text{max}$ the motor stalls; above it the motor runs at
   target. The transition characterises the system's "stall
   tension" — a useful design parameter.

---

## 11. Slot in the broader plan

This plan is independent of `PLAN_feeder_strategy_overview.md` and its
three branches: any of those feeder models can be combined with this
downstream change. We recommend landing torque-limited Roll B **first**
(it is simpler) so that the feeder-branch validation suite
(strategy §5) sees a properly-bounded downstream and doesn't conflate
feeder-side compliance issues with the unbounded-Roll-B artefact.
