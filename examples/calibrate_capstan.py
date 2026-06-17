#!/usr/bin/env python3
"""examples/calibrate_capstan.py — identify guide friction from experimental
tension and drive the sim to reproduce it.

Pipeline (see examples/CAPSTAN_VALIDATION_PLAN.md):

    experiment T_A,T_B(t)
      → identify μ via the capstan equation
      → write a calibrated params JSON
      → run the sim headless (run_headless in yarn_rolls_ogc_gui)
      → compare sim vs experiment.

Capstan (Eytelwein):   T_tight = T_slack · exp(μ · β)
  ⇒   μ = ln(T_tight / T_slack) / β        (β = wrap angle; ~π/2 for 90°)

Roll B pulls (motion A→B), friction opposes motion ⇒ the DOWNSTREAM side (B) is
tight, so normally T_B > T_A.  The ratio is independent of yarn stiffness, so μ
is recovered cleanly even with stiffness unknown; the yarn slides continuously,
so this μ is the KINETIC coefficient → set guide_mu_k = μ (and guide_mu_s ≈ μ).

This module is pure-stdlib for everything except `run` (which lazily imports the
Warp sim).  So `selftest`, `identify`, `params`, and `compare` run anywhere.

Subcommands:
    selftest                      prove μ round-trips on synthetic data (no sim)
    identify  EXP.csv             print μ from experimental steady means
    params    EXP.csv BASE.json   identify μ, write a calibrated params JSON
    run       PARAMS.json         run the sim headless → sim CSV   (needs Warp)
    compare   EXP.csv SIM.csv     metrics: ratio error, mean error, μ match
    auto      EXP.csv BASE.json   params → run → compare           (needs Warp)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import tempfile


# ── Capstan core ──────────────────────────────────────────────────────────────

def capstan_mu(t_tight: float, t_slack: float, beta: float) -> float:
    """μ from a tight/slack tension pair over wrap angle β (radians)."""
    if t_tight <= 0.0 or t_slack <= 0.0:
        raise ValueError(f"tensions must be > 0 (got tight={t_tight}, slack={t_slack})")
    if beta <= 0.0:
        raise ValueError(f"wrap angle β must be > 0 (got {beta})")
    return math.log(t_tight / t_slack) / beta


def identify_mu(t_a: float, t_b: float, beta: float):
    """Return (μ, tight_side_label). Tight side = the larger tension."""
    if t_b >= t_a:
        return capstan_mu(t_b, t_a, beta), "B (downstream)"
    return capstan_mu(t_a, t_b, beta), "A (upstream)"


def capstan_ratio(mu: float, beta: float) -> float:
    """Predicted T_tight / T_slack for a given μ and wrap angle."""
    return math.exp(mu * beta)


# ── Data IO ─────────────────────────────────────────────────────────────────--

def _fmean(xs) -> float:
    xs = list(xs)
    return (sum(xs) / len(xs)) if xs else float("nan")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


# header-name aliases → role
_T_TIME = {"t", "time", "times", "sec", "secs", "second", "seconds", "ts"}
_T_A = {"ta", "tensiona", "ya", "ina", "upstream", "tin", "tslack", "slack"}
_T_B = {"tb", "tensionb", "yb", "outb", "downstream", "tout", "ttight", "tight"}


def _resolve_col(spec, header):
    """spec may be an int-like index, a column name, or None. header may be None."""
    if spec is None:
        return None
    s = str(spec)
    if s.lstrip("+-").isdigit():
        return int(s)
    if header is not None:
        want = _norm(s)
        for i, h in enumerate(header):
            if _norm(h) == want:
                return i
    raise ValueError(f"column {spec!r} not found in header {header}")


def load_tension_csv(path, t_col=None, a_col=None, b_col=None):
    """Load (t, T_A, T_B) from a CSV. Robust to header/no-header and column order.

    Autodetects columns named like time / T_A / T_B; otherwise assumes
    [t, T_A, T_B] (3+ cols) or [T_A, T_B] (2 cols).  t may be returned as None.
    Explicit *_col (index or name) override autodetection.
    """
    with open(path, newline="") as f:
        rows = [r for r in csv.reader(f) if r and any(c.strip() for c in r)]
    if not rows:
        raise ValueError(f"{path}: empty")

    # Header present if the first row has a non-numeric cell.
    def _is_num(x):
        try:
            float(x)
            return True
        except ValueError:
            return False
    has_header = not all(_is_num(c) for c in rows[0])
    header = rows[0] if has_header else None
    data = rows[1:] if has_header else rows
    ncol = len(rows[0])

    ti = _resolve_col(t_col, header)
    ai = _resolve_col(a_col, header)
    bi = _resolve_col(b_col, header)

    if header is not None and (ai is None or bi is None):
        for i, h in enumerate(header):
            n = _norm(h)
            if ai is None and n in _T_A:
                ai = i
            elif bi is None and n in _T_B:
                bi = i
            elif ti is None and n in _T_TIME:
                ti = i

    if ai is None or bi is None:                 # positional fallback
        if ncol >= 3:
            ti, ai, bi = (ti if ti is not None else 0), 1, 2
        elif ncol == 2:
            ai, bi = 0, 1
        else:
            raise ValueError(f"{path}: need ≥2 columns, got {ncol}")

    t, ta, tb = [], [], []
    for r in data:
        try:
            ta.append(float(r[ai]))
            tb.append(float(r[bi]))
            t.append(float(r[ti]) if ti is not None else float(len(t)))
        except (ValueError, IndexError):
            continue            # skip malformed rows
    if not ta:
        raise ValueError(f"{path}: no numeric rows parsed")
    return t, ta, tb


def steady_means(ta, tb, frac=0.5):
    """Mean of T_A, T_B over the last `frac` of the series (steady state)."""
    n = len(ta)
    k = max(1, int(round(n * frac)))
    return _fmean(ta[-k:]), _fmean(tb[-k:]), k


def synthetic_capstan_csv(path, mu, beta, t_a_level=10.0, n=400, noise=0.0,
                          seed=0, dt=0.01):
    """Write a fake experiment where T_B = T_A·exp(μβ) (+ optional noise)."""
    import random
    rng = random.Random(seed)
    ratio = capstan_ratio(mu, beta)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "T_A", "T_B"])
        for i in range(n):
            jitter_a = noise * rng.uniform(-1.0, 1.0) if noise else 0.0
            jitter_b = noise * rng.uniform(-1.0, 1.0) if noise else 0.0
            ta = t_a_level + jitter_a
            tb = ta * ratio + jitter_b
            w.writerow([f"{i*dt:.4f}", f"{ta:.6f}", f"{tb:.6f}"])
    return path


# ── Params writer ───────────────────────────────────────────────────────────--

def write_params(base_path, out_path, mu, guide_static_scale=1.0, overrides=None):
    """Load a base params JSON, set the identified guide friction, write a copy.

    `guide_mu_k` = μ (kinetic, the sliding capstan coefficient);
    `guide_mu_s` = μ · guide_static_scale (≥ μ).  `overrides` patches any other
    keys (geometry / feed / pull) the caller wants pinned from the experiment.
    """
    with open(base_path) as f:
        data = json.load(f)
    data["guide_mu_k"] = float(mu)
    data["guide_mu_s"] = float(mu) * float(guide_static_scale)
    if overrides:
        data.update(overrides)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    return out_path


# ── Comparison ────────────────────────────────────────────────────────────────

def compare(exp_csv, sim_csv, beta, frac=0.5):
    te, tae, tbe = load_tension_csv(exp_csv)
    ts, tas, tbs = load_tension_csv(sim_csv)
    ea, eb, ke = steady_means(tae, tbe, frac)
    sa, sb, ks = steady_means(tas, tbs, frac)
    re_ = eb / ea if abs(ea) > 1e-12 else float("nan")
    rs_ = sb / sa if abs(sa) > 1e-12 else float("nan")
    mu_e = identify_mu(ea, eb, beta)[0]
    mu_s = identify_mu(sa, sb, beta)[0]
    print("── compare (steady-state means) ──────────────────────────────")
    print(f"  experiment:  T_A={ea:8.3f}  T_B={eb:8.3f}  T_B/T_A={re_:.4f}  μ={mu_e:.4f}")
    print(f"  simulation:  T_A={sa:8.3f}  T_B={sb:8.3f}  T_B/T_A={rs_:.4f}  μ={mu_s:.4f}")
    if abs(ea) > 1e-12 and abs(eb) > 1e-12:
        print(f"  mean error:  T_A={100*(sa-ea)/ea:+6.2f}%   T_B={100*(sb-eb)/eb:+6.2f}%")
    if re_ == re_ and rs_ == rs_ and abs(re_) > 1e-12:
        print(f"  ratio error: {100*(rs_-re_)/re_:+6.2f}%   (Δμ={mu_s-mu_e:+.4f})")
    print("──────────────────────────────────────────────────────────────")
    return {"mu_exp": mu_e, "mu_sim": mu_s, "ratio_exp": re_, "ratio_sim": rs_}


# ── Self-test (no sim) ──────────────────────────────────────────────────────--

def selftest():
    beta = math.pi / 2.0
    print(f"β = 90° = {beta:.6f} rad")
    tmp = tempfile.mkdtemp(prefix="capstan_selftest_")
    ok = True

    # 1) Exact round-trip: synth with known μ → recover it to machine precision.
    for mu_true in (0.05, 0.12, 0.26, 0.40):
        p = os.path.join(tmp, f"synth_{mu_true}.csv")
        synthetic_capstan_csv(p, mu_true, beta, t_a_level=10.0, n=400, noise=0.0)
        t, ta, tb = load_tension_csv(p)
        a, b, k = steady_means(ta, tb)
        mu_rec, tight = identify_mu(a, b, beta)
        err = abs(mu_rec - mu_true)
        # tolerance is set by the synthetic CSV's %.6f write precision (~1e-8 in μ)
        flag = "ok" if err < 1e-6 else "FAIL"
        ok = ok and err < 1e-6
        print(f"  μ_true={mu_true:.3f}  T_B/T_A={b/a:.4f}  μ_rec={mu_rec:.6f}  "
              f"|Δ|={err:.2e}  tight={tight}  [{flag}]")

    # 2) Noisy: recover μ within a loose tolerance from the steady means.
    mu_true = 0.20
    p = os.path.join(tmp, "synth_noisy.csv")
    synthetic_capstan_csv(p, mu_true, beta, t_a_level=10.0, n=2000, noise=0.5, seed=7)
    t, ta, tb = load_tension_csv(p)
    a, b, _ = steady_means(ta, tb)
    mu_rec = identify_mu(a, b, beta)[0]
    err = abs(mu_rec - mu_true)
    flag = "ok" if err < 0.02 else "FAIL"
    ok = ok and err < 0.02
    print(f"  noisy: μ_true={mu_true:.3f}  μ_rec={mu_rec:.4f}  |Δ|={err:.4f}  [{flag}]")

    # 3) Tight-side symmetry: swapping A/B yields the same μ.
    mu_ab = identify_mu(10.0, 14.0, beta)[0]
    mu_ba = identify_mu(14.0, 10.0, beta)[0]
    flag = "ok" if abs(mu_ab - mu_ba) < 1e-12 else "FAIL"
    ok = ok and abs(mu_ab - mu_ba) < 1e-12
    print(f"  symmetry: μ(10,14)={mu_ab:.6f}  μ(14,10)={mu_ba:.6f}  [{flag}]")

    print("SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


# ── CLI ───────────────────────────────────────────────────────────────────────

def _beta(args):
    return math.radians(args.beta_deg)


def cmd_identify(args):
    t, ta, tb = load_tension_csv(args.exp, args.t_col, args.a_col, args.b_col)
    a, b, k = steady_means(ta, tb, args.frac)
    mu, tight = identify_mu(a, b, _beta(args))
    print(f"steady means over last {k} samples:")
    print(f"  T_A={a:.4f}  T_B={b:.4f}  T_B/T_A={b/a:.4f}  (tight side: {tight})")
    print(f"  β={args.beta_deg:.1f}°  →  μ = {mu:.5f}")
    return 0


def cmd_params(args):
    t, ta, tb = load_tension_csv(args.exp, args.t_col, args.a_col, args.b_col)
    a, b, k = steady_means(ta, tb, args.frac)
    mu, tight = identify_mu(a, b, _beta(args))
    out = args.out or os.path.splitext(args.exp)[0] + "-calibrated-params.json"
    write_params(args.base, out, mu, guide_static_scale=args.static_scale)
    print(f"identified μ={mu:.5f} (tight: {tight}); wrote {out}")
    print(f"  set guide_mu_k={mu:.5f}, guide_mu_s={mu*args.static_scale:.5f}")
    return 0


def cmd_run(args):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from yarn_rolls_ogc_gui import run_headless          # lazy: needs Warp
    out = run_headless(args.params, seconds=args.seconds, out_csv=args.out)
    print(f"sim CSV → {out}")
    return 0


def cmd_compare(args):
    compare(args.exp, args.sim, _beta(args), args.frac)
    return 0


def cmd_auto(args):
    t, ta, tb = load_tension_csv(args.exp, args.t_col, args.a_col, args.b_col)
    a, b, k = steady_means(ta, tb, args.frac)
    mu, tight = identify_mu(a, b, _beta(args))
    params_out = args.out or os.path.splitext(args.exp)[0] + "-calibrated-params.json"
    write_params(args.base, params_out, mu, guide_static_scale=args.static_scale)
    print(f"identified μ={mu:.5f} (tight: {tight}); wrote {params_out}")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from yarn_rolls_ogc_gui import run_headless          # lazy: needs Warp
    sim_csv = run_headless(params_out, seconds=args.seconds)
    compare(args.exp, sim_csv, _beta(args), args.frac)
    return 0


def _add_cols(p):
    p.add_argument("--t-col", default=None, help="time column (name or index)")
    p.add_argument("--a-col", default=None, help="T_A column (name or index)")
    p.add_argument("--b-col", default=None, help="T_B column (name or index)")


def _add_beta_frac(p):
    p.add_argument("--beta-deg", type=float, default=90.0, help="wrap angle (deg)")
    p.add_argument("--frac", type=float, default=0.5,
                   help="fraction of the tail used for steady-state means")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("selftest", help="round-trip the capstan math on synthetic data")

    p = sub.add_parser("identify", help="print μ from experimental tension")
    p.add_argument("exp"); _add_cols(p); _add_beta_frac(p)

    p = sub.add_parser("params", help="identify μ and write a calibrated params JSON")
    p.add_argument("exp"); p.add_argument("base")
    p.add_argument("-o", "--out", default=None)
    p.add_argument("--static-scale", type=float, default=1.0,
                   help="guide_mu_s = μ · scale (≥1)")
    _add_cols(p); _add_beta_frac(p)

    p = sub.add_parser("run", help="run the sim headless from a params JSON")
    p.add_argument("params"); p.add_argument("-o", "--out", default=None)
    p.add_argument("--seconds", type=float, default=10.0)

    p = sub.add_parser("compare", help="compare experiment vs sim CSV")
    p.add_argument("exp"); p.add_argument("sim"); _add_beta_frac(p)

    p = sub.add_parser("auto", help="params → run → compare")
    p.add_argument("exp"); p.add_argument("base")
    p.add_argument("-o", "--out", default=None)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--static-scale", type=float, default=1.0)
    _add_cols(p); _add_beta_frac(p)

    args = ap.parse_args(argv)
    return {
        "selftest": lambda a: selftest(),
        "identify": cmd_identify,
        "params":   cmd_params,
        "run":      cmd_run,
        "compare":  cmd_compare,
        "auto":     cmd_auto,
    }[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
