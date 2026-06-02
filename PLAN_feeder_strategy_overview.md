# Strategy: Faithful Yarn-Feeder Model — Overview

Status: planning. No code changes yet. This document frames the three
parallel branches we are about to explore and ties them to the existing
servo-control work.

---

## 1. The diagnosis

Two physically distinct elements in the real EFS-32 + package setup are
conflated into our single `Roll A`:

| Real element       | Function                              | Yarn on it    | Compliance      |
|--------------------|---------------------------------------|---------------|-----------------|
| Bobbin / package   | Stores bulk yarn, passive payout      | hundreds of m | High (stretchy) |
| EFS-32 drum        | Motor-actuated drive interface        | 3–8 wraps     | ~zero (Capstan) |

In our current model `Roll A` carries ~95 % of all particles wound around
it (with `yarn_length = 50 m` the helix essentially is the whole yarn).
Those wound particles are **free** under OGC contact, so the entire spool
behaves as one large elastic accumulator. That is the wrong compliance
distribution. The real machine isolates the downstream control loop from
the bulk-yarn elasticity by the exponential Capstan amplification on the
drum wraps.

Consequence: the servo measures and acts on a plant whose dynamic stiffness
is dominated by physics that isn't supposed to exist (~50 m of compliant
yarn elastically connected through the drum). Any controller — PI or the
EFS-32 tension-window — will misbehave on this plant.

---

## 2. Three branches

We will prototype three remedies in parallel branches. They share the
diagnosis but differ in how much physical fidelity they add.

| #  | Branch name (proposed)        | Idea (one line)                                                                 | Plan doc |
|----|-------------------------------|---------------------------------------------------------------------------------|----------|
| O1 | `feeder/o1-kinematic-drum`    | Short wound section, all wound particles kinematically driven by drum rotation. | [`PLAN_feeder_option1_kinematic_drum.md`](PLAN_feeder_option1_kinematic_drum.md) |
| O2 | `feeder/o2-drum-plus-package` | O1 plus an upstream compliant tail standing in for package payout.              | [`PLAN_feeder_option2_drum_plus_package.md`](PLAN_feeder_option2_drum_plus_package.md) |
| O3 | `feeder/o3-two-cylinder`      | Separate **package cylinder** + **EFS drum cylinder**; full bulk-yarn modelled. | [`PLAN_feeder_option3_two_cylinder.md`](PLAN_feeder_option3_two_cylinder.md) |

The three branches are **subset-supersets**, not alternatives:

```
O1  ⊂  O2  ⊂  O3
```

— in the sense that O2 retains everything in O1 and adds upstream
compliance, and O3 makes the bulk yarn an explicit physical object on its
own cylinder. The minimum useful change is O1; the maximum-fidelity
change is O3. Building all three lets us measure, for the same controller,
which compliances actually matter.

---

## 3. Where these branches sit relative to the other plans

The feeder-model branches (O1–O3) change the **upstream plant**. They
sit alongside three other plans that change the **controller** and the
**downstream plant**:

| Plan                                         | Scope                          | Status        |
|----------------------------------------------|--------------------------------|---------------|
| `PLAN_tension_servo_autowarp.md`             | Controller (Phase 3 PI servo)  | merged        |
| `PLAN_collocated_anchor_feedback.md`         | Controller (where it measures) | planned       |
| `PLAN_roll_b_torque_limit.md`                | Downstream plant (Roll B)      | merged (913e82c) |
| *(forthcoming)* `PLAN_efs32_tension_window.md` | Controller (EFS-32 law)      | sketched only |
| **This document → O1, O2, O3**               | **Upstream plant (Roll A)**    | **planned**   |

The plans are orthogonal in scope but compose cleanly. The composition
rules:

1. Anchor feedback (`PLAN_collocated_anchor_feedback.md`) is **shared** —
   every feeder branch should land on top of it. The "anchor segment" in
   O1/O2/O3 is defined slightly differently (see each plan), but the
   formula and the GUI surface are the same.
2. Roll B torque limit (`PLAN_roll_b_torque_limit.md`) is **shared** —
   lands once on master, in front of any feeder work. With Roll B as a
   bounded actuator, the feeder-branch validation suite (§5) measures
   feeder-side compliance honestly instead of being polluted by Roll B
   stretching the yarn unbounded.
3. EFS-32 tension-window controller is **shared** — once landed, it
   replaces PI on every branch. Each branch should report tension-band
   limits in physically meaningful cN regardless of feeder model.
4. The auto-warp geometry helper (`_warp_keypoints`,
   `make_initial_positions`) keeps the same external interface across
   branches — the wound-section length changes, but the keypoint
   computation does not.

---

## 4. Common terminology

To keep the three plans consistent:

- **Drum** — the cylinder driven by the motor (`Roll A` today).
- **Package** — the upstream bulk-yarn source. Implicit in O1, a soft
  anchor in O2, an explicit second cylinder in O3.
- **Drum wraps** ($n_w$) — number of full revolutions of yarn wound on
  the drum. Geometric input; particle count is derived.
- **Drum-wound particles** — the set $\{i \mid i < n_\text{wound}\}$
  where $n_\text{wound}$ is computed from $n_w$ and the OGC pitch.
- **Anchor segment** (for the collocated feedback plan) — the first
  free-span segment leaving the drum, i.e. segment
  $[\,n_\text{wound} - 1,\; n_\text{wound}\,]$ in O1/O2/O3, **not**
  $[\,0, 1\,]$ as in today's code. (This is the one paragraph in
  `PLAN_collocated_anchor_feedback.md` that needs updating once any of
  these feeder branches lands — flagged here so it isn't missed.)

---

## 5. Validation plan (shared by all three branches)

Each branch runs the same battery of tests, with the controller held
fixed (initially PI; later EFS-32). We compare:

### 5.1 Plant identification (open loop)

1. **Step in motor speed.** Hold tension setpoint inactive, drive
   $\omega_A$ from 0 → $\omega_\text{set}$ at $t=1$ s. Record
   anchor-tension response. Read off:
   - Rise time
   - Steady-state tension
   - First overshoot (if any)
2. **Step in downstream pull.** Hold $\omega_A = 0$, jump
   `pull_speed` from 0 → $v_\text{pull}$. Record anchor-tension
   propagation latency from when downstream starts pulling to when
   anchor segment registers extension.

Expectation across branches:
- O1: minimal compliance, fast rise, no ringing.
- O2: similar to O1 plus a damped low-frequency mode from the upstream
  spring.
- O3: O2 behaviour plus measurable Capstan attenuation from drum to
  package (the upstream side should be visibly decoupled).

### 5.2 Closed-loop response (PI)

1. **Setpoint step.** Setpoint 5 → 10 cN at $t=2$ s, identical
   $k_p, k_i$. Measure overshoot, settle time, residual error.
2. **Stability boundary.** Sweep $k_p$ upward until oscillation appears.
   Record the gain margin.

Expected ordering: O3 ≳ O2 > O1 in stability margin (more compliance
upstream → slower disturbances reach the controller → less effective
delay), but O1 simplest to tune.

### 5.3 Capstan residual

For each branch: enable downstream pull, observe steady-state
$T_B / T_A$ and $\rho = T_B / (T_A e^{\mu_k \theta})$.

- O1: kinematic helix has no Capstan effect on the drum *by
  construction*; residual should reflect only the guide-cylinder
  Capstan.
- O2: same as O1; spring-anchor adds no friction surface.
- O3: package cylinder may show its own Capstan contribution.

---

## 6. Sequencing recommendation

We don't need to build all three at the same time, even though they will
end up on separate branches.

1. **First land** `PLAN_roll_b_torque_limit.md` on `master`. This caps
   downstream behaviour so the upstream branches' validation suite (§5)
   isn't polluted by Roll B stretching the yarn without limit. It is
   independent of the feeder model.
2. **Then land** `PLAN_collocated_anchor_feedback.md` on `master`. This
   is feeder-agnostic and unlocks meaningful upstream tension readings
   for the validation in §5.
3. **Build O1** next. It is the smallest physically faithful upstream
   step, and it is a prerequisite for O2. ETA: a few hours.
4. **Build O2** as a continuation of O1 on its own branch. Most of the
   O1 code is reused.
5. **Build O3** in parallel, on its own branch off `master` (not off
   O2), because the two-cylinder geometry diverges enough that diffing
   back onto O1/O2 will be painful otherwise.
6. **Compare** all three under the same controller and the same
   validation suite (§5). Merge the best back into `master`. Keep the
   others on their branches for future use.

---

## 7. What this overview deliberately does *not* decide

- **The EFS-32 controller law** (tension-band, proportional inside the
  band, hard zeros outside). That lives in its own forthcoming plan.
- **Whether to model an explicit knitting-side tension dancer.** Roll B
  remains a constant-speed motor for now. The tension on the downstream
  side is whatever the simulation produces.
- **The auto-warp helix-pitch parameter.** Currently fixed at one OGC
  diameter per turn; can be exposed as a slider later if pitch matters
  for any branch.
