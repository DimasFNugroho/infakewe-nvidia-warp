# Capstan Validation Plan — matching the sim to real yarn experiments

**Status:** in progress — headless runner + `calibrate_capstan.py` scaffolding landed & self-tested (data-independent); awaiting experimental data
**Owner:** Dimas
**Created:** 2026-06-17
**Sim under test:** `examples/yarn_rolls_ogc_gui.py` (Warp + OGC, vispy GUI)

---

## 1. Objective

Build a **separate calibration script** that:

1. reads experimental tension data `T_A(t)`, `T_B(t)`,
2. uses the **capstan equation** to back out the unknown **guide friction** μ, and
3. emits a **params JSON** that drives our existing sim so it reproduces the experiment,
4. then runs the sim **headless** and reports `T_A`/`T_B` back for comparison.

The script's role is to **produce input to the simulation** and to validate that the
simulation reproduces the measured behaviour of the yarn ↔ guiding-element interaction
(surface friction, tension, wrap angle; vibration deferred — see §9).

---

## 2. Experimental setup (as described)

- Chain: **package → Roll A (feeder) → guide → Roll B (puller) → vacuum (free tail)**.
- Measured: **`T_A(t)` and `T_B(t)`** only. Other parameters are constants.
- **Friction constants are unknown** — this is what we identify.
- **Wrap ("warp") angle ≈ 90°** on the guide ⇒ `β = π/2 rad`.
- Roll B is a constant-speed motor (pulls); yarn after it is disposed (free end).

> Sensor placement assumption (to confirm): **Sensor A upstream** of the guide,
> **Sensor B downstream** of the guide, so the two sensors straddle **only** the guide.

---

## 3. Physical model — the capstan (Eytelwein) equation

Yarn slides over the (non-rotating) guide while Roll B pulls it. Across the guide:

```
T_tight = T_slack · exp(μ · β)
```

- `β` = wrap angle ≈ 90° = π/2 rad.
- Motion is A→B (Roll B pulls); friction opposes motion ⇒ **downstream side (B) is tight**.
  Expect `T_B > T_A`.
- Invert for the unknown friction:

```
μ = ln(T_B / T_A) / β  =  (2/π) · ln(T_B / T_A)        [β = π/2]
```

**Why this is well-posed:**

- The **ratio `T_B/T_A` is independent of yarn stiffness/modulus** — so we recover μ
  cleanly even though stiffness *and* friction are both unknown.
- The yarn slides continuously ⇒ recovered μ is the **kinetic** coefficient
  ⇒ set `guide_mu_k ≈ μ`, `guide_mu_s ≈ μ` (or slightly higher).

**Centrifugal correction (refinement, only if yarn is fast):**

```
T_tight − ρ·v²  =  (T_slack − ρ·v²) · exp(μ·β)
```
where ρ = yarn linear density (kg/m), v = yarn speed (m/s). Negligible at low speed;
fold in once speed + linear density are known.

---

## 4. Key findings from the existing sim (de-risk the build)

**Finding 1 — the sim already embeds the capstan equation.**
`_write_shared()` (≈ `yarn_rolls_ogc_gui.py:1223`) already computes, every frame:

```python
theta, n_contact = _wrap_angle_contacts(pp)        # ACTUAL wrap angle from contacts
mu_k  = state["guide_mu_k"]
capstan_pred = T_a * np.exp(mu_k * theta)
residual     = T_b / capstan_pred                  # ≈ 1 ⟺ sim friction matches capstan ratio
# shared layout: [T_a, T_b, theta_deg, capstan_pred, residual, sim_time, n_contact]
```

⇒ The forward sim **self-reports** how well its guide friction reproduces the capstan
relation, and it **measures the real wrap angle θ** (so we can verify it is ~90°).
**Calibration target becomes crisp:** choose `guide_mu_k` so the sim's `residual ≈ 1`
*and* its `T_A`/`T_B` means match the experiment.

**Finding 2 — headless is a surgical refactor, not a rewrite.**
- Tension measurement `_write_shared(pp, sim_t)` is **already UI-free** (positions in,
  `shared[]` out).
- Physics stepping is in `_execute_substeps()` (`:1291`).
- Only the per-frame **control** logic (Roll A feeder/servo, `_set_*_omega`, `sim_time`
  advance) is tangled inside `on_timer()` (`:2246`) next to rendering.
- ⇒ Extract that into a UI-free `_frame_update()` that **both** `on_timer` and a headless
  loop call ⇒ headless physics is **identical** to the GUI (no divergence).

Relevant code map:
| Symbol | Location | Role |
|---|---|---|
| `sim_worker` | `:174` | builds Warp+OGC+vispy in one process |
| `_execute_substeps` | `:1291` | PBD/OGC physics step |
| `_tension_from_mask` | `:1018` | avg tension (cN) over masked particles |
| `_write_shared` | `:1223` | T_A/T_B/θ/capstan_pred/residual → `shared[]` |
| `_wrap_angle_contacts` | (called `:1246`) | measured wrap angle θ |
| `on_timer` | `:2246` | per-frame control + render + `app.run()` driver |
| `run_ui` | `:2517` | tkinter control panel |

---

## 5. Decisions locked (2026-06-17)

| Decision | Choice | Implication |
|---|---|---|
| Automation level | **Params + headless check** | build a headless runner; user tweaks from there (no auto-optimizer yet) |
| What to match first | **Steady-state means** | match average `T_A`/`T_B` + capstan ratio (→ μ); defer vibration |

---

## 6. Pipeline

```
experiment T_A,T_B(t)
   │  steady means  T̄_A, T̄_B
   ▼
[identify μ]  μ = ln(T̄_B/T̄_A)/β
   ▼
[write params.json]  guide_mu_s/k=μ, 90° wrap geometry, feed/pull/yarn props
   ▼
[run_headless]  loop: _execute_substeps → _write_shared → _frame_update  → CSV
   ▼
[compare]  sim T_A,T_B,residual  vs  experiment  → overlay plot + error metrics
   ▼
(manual μ / level tweak; closed-loop auto-fit is future work §9)
```

---

## 7. Staged steps & deliverables

### Phase 0 — Lock data & geometry  *(needs user input)*
- [ ] Data format/units/sample-rate/duration confirmed.
- [ ] Confirm Sensor A upstream / Sensor B downstream of guide (straddle only the guide).
- [ ] Confirm β ≈ 90° and that yarn is moving (kinetic regime) during measurement.
- [ ] Collect constants (see §8).

### Phase 1 — Capstan identification *(core, data-dependent)*
- [ ] Compute steady means `T̄_A`, `T̄_B` (windowing/outlier handling).
- [ ] Determine tight side from data (expect `T_B > T_A`).
- [ ] Output `μ_s`, `μ_k` + residual/confidence.
- **Deliverable:** printed μ + a `*_identified.json` fragment.

### Phase 2 — Params generation *(script's main output, data-dependent)*
- [ ] Map constants + μ → full params JSON (existing sim format).
  - guide geometry placed for 90° wrap; `guide_mu_s/k = μ`; guide radius/material.
  - Roll A feeder tuned to the measured `T_A` level.
  - Roll B `pull_speed` = measured yarn speed.
  - yarn segment/stiffness from linear density/modulus.
- **Deliverable:** `examples/<exp-name>-calibrated-params.json`.

### Phase 3 — Headless runner *(data-independent — DONE except parity check)*
- [x] Extract `_frame_update(pp)` from `on_timer` (GUI still calls it; returns
      the HUD scalars).
- [x] `sim_worker(headless=True, headless_seconds, headless_out)` early-returns
      before vispy; `_run_headless()` loops `sim_step`+`_write_shared`+
      `_frame_update` → CSV `t,T_A,T_B,theta_deg,capstan_pred,residual` + a
      steady-state summary.
- [x] Module-level `run_headless(params_path, seconds, out_csv)` entry point
      (merges a params JSON onto `DEFAULTS`, single-process, no vispy).
- [ ] Sanity (needs Warp env): GUI and headless give the same `T_A`/`T_B` for
      the same params. **← run on your machine.**
- **Deliverable:** headless entry point + CSV. ✅ (parity check pending)

### Phase 4 — Comparison & report *(data-dependent)*
- [ ] Overlay sim vs experiment `T_A(t)`, `T_B(t)`.
- [ ] Metrics: mean error per channel; **ratio error** `|(T_B/T_A)_sim − (T_B/T_A)_exp|`;
      sim `residual` proximity to 1; measured θ vs 90°.
- **Deliverable:** comparison plot(s) + metrics printout.

### Phase 5 — `calibrate_capstan.py` glue *(ties 1–4 together — DONE)*
- [x] CLI subcommands: `selftest`, `identify EXP.csv`, `params EXP.csv BASE.json`,
      `run PARAMS.json`, `compare EXP.csv SIM.csv`, `auto EXP.csv BASE.json`.
- [x] Robust CSV loader (header/no-header, column autodetect + `--a-col/--b-col`
      overrides), steady-state means over the tail (`--frac`), `--beta-deg`.
- [x] Pure-stdlib for identify/params/compare/selftest (runs without Warp);
      `run`/`auto` lazily import `run_headless`.
- [x] Prove μ **round-trips** on synthetic CSVs — `selftest` PASSES (exact to
      ~1e-8; noisy μ=0.20→0.1999; tight-side symmetry). Verified `identify`+
      `params`+`compare` end-to-end on synthetic data against `params-main.json`.
- **Deliverable:** one script that runs the whole pipeline. ✅

---

## 8. Data & constants needed from experiment

- [ ] Data file(s): columns, **units** (cN?), sample rate, duration.
- [ ] Yarn **linear density** (tex / dtex / g/m).
- [ ] **Yarn speed** (m/min or m/s).
- [ ] **Guide** diameter + material.
- [ ] Roll A & Roll B radii / speeds.
- [ ] Any tension setpoints used.
- [ ] Yarn **modulus / denier** if known (for absolute-level matching; not needed for μ).
- [ ] The graphs referenced.

---

## 9. Success criteria

- `μ` recovered with a physically plausible value (yarn-on-ceramic/metal ≈ 0.1–0.4).
- Sim **steady** `T_B/T_A` matches experiment within target tolerance (TBD, e.g. ≤5–10%).
- Sim `residual ≈ 1`; measured wrap θ ≈ 90°.
- (Stretch) absolute `T_A`, `T_B` levels matched after feed/pull/stiffness tuning.

---

## 10. Future work (explicitly deferred)

- **Vibration / dynamics:** FFT experimental tension → dominant freq/amplitude →
  inject as guide/tension perturbation → match spectra.
- **Closed-loop auto-fit:** optimizer that runs `run_headless` repeatedly and auto-tunes
  μ (and feed/pull/`stretch_stiff`) to minimize error.
- **Centrifugal correction** (§3) if yarn speed is high.

---

## 11. Files

| File | Status | Purpose |
|---|---|---|
| `examples/CAPSTAN_VALIDATION_PLAN.md` | this doc | the plan / tracker |
| `examples/yarn_rolls_ogc_gui.py` | ✅ edited | `_frame_update()` extracted; `run_headless()` added |
| `examples/calibrate_capstan.py` | ✅ created | identify μ, write params, run+compare, selftest |
| `examples/<exp>-calibrated-params.json` | to generate | sim input from experiment |

### Usage (run from `examples/`)

```bash
# 0. prove the math (no Warp needed) — passes today
python3 calibrate_capstan.py selftest

# 1. when data lands: read μ from the experiment (β defaults to 90°)
python3 calibrate_capstan.py identify  EXP.csv
#    columns autodetected; override with --a-col/--b-col (name or index),
#    --beta-deg, --frac (tail fraction used for the steady mean)

# 2. write a calibrated params JSON from a known-good base (e.g. 04 / params-main)
python3 calibrate_capstan.py params    EXP.csv params-main.json -o EXP-calibrated-params.json

# 3. run the sim headless (needs Warp/GPU) → sim CSV
python3 calibrate_capstan.py run       EXP-calibrated-params.json --seconds 10

# 4. compare experiment vs sim
python3 calibrate_capstan.py compare   EXP.csv EXP-calibrated-params-sim.csv

# or do 2→3→4 in one shot:
python3 calibrate_capstan.py auto      EXP.csv params-main.json --seconds 10
```

---

## 12. Open questions / assumptions

- Sensor A/B really straddle only the guide (no other frictional contact between them)?
- Guide is **fixed** (yarn slides) vs a roller (would change the friction model)?
- Tension units in the data (cN assumed, matching the sim HUD)?
- Is `T_B > T_A` in the data (confirms tight side = downstream)?
