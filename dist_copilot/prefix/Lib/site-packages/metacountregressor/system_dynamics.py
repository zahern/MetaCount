"""
system_dynamics.py
================================================================================
System-dynamics (stock-flow) module with genetic-programming form discovery.

This packages the game-day stock-flow *twin* that used to live inline in the
Suncorp pipeline (``scripts/59_game_day_search_predict.py``) so it is a
first-class, reusable part of metacountregressor and can be driven by the same
ExperimentBuilder workflow as the statistical structure search.

The loop
--------
Traffic flow ``q`` is an exogenous driver.  It is converted to density
(``k = q / v``), density sets speed through a speed-density relation ``v(k)``,
speed minus the posted limit gives exceedance ``e``, and exceedance maps to a
speeding rate ``lambda(e)``.  Predicted counts are the product ``q * rate``:

    q  ->  k = q/v  ->  v(k)  ->  e = v - limit  ->  lambda(e)  ->  y = q*lambda

``k`` is the *stock* (vehicles per unit road per unit time); ``q`` is the
*flow*.  A game changes ``q`` (demand), never the curves directly -- that is
what makes the twin answer "how much of the game-day speeding change is
density-mediated?" rather than just re-fitting the same regression.

Two ways to get the shapes
--------------------------
1. **Theory (analytic)**: Greenshields ``v = vf*(1 - k/kj)`` with ``kj`` from
   road geometry (jam density is physics, not a statistical parameter) and only
   ``vf`` estimated; a smoothed empirical rate table for ``lambda(e)``.
2. **Genetic program**: a bounded symbolic-regression search over the binned
   ``(k, v)`` and ``(e, log-rate)`` curves.  The GP is *seeded* with the theory
   forms and keeps the elite best-so-far, so a discovered form can only
   tie-or-beat the analytic baseline -- never lose to it.  A time budget bounds
   each run; on timeout/failure the analytic form stands.

Both paths share the exact same forward simulation (``sd_predict``), including
the fixed-point between ``k`` and ``v`` (capped at 3 rounds) and hard clipping,
so a wild discovered form degrades to the parameter bounds instead of blowing
up to NaN.

Zero new dependencies: numpy + pandas only.
"""
from __future__ import annotations

import time
import zlib

import numpy as np
import pandas as pd

__all__ = [
    "SD_EDGES",
    "gp_discover_form",
    "fit_speed_density",
    "fit_propensity_lookup",
    "fit_sd_forms",
    "sd_regime",
    "sd_predict",
    "SystemDynamicsModel",
    "fit_system_dynamics",
]

# Exceedance (observed speed - limit) bin edges, km/h.  Shared by the empirical
# rate table and the GP binned-rate target so fit and predict agree on bins.
SD_EDGES = np.linspace(-30.0, 60.0, 19)


def _crc(s) -> int:
    """Stable per-class seed offset (Python's hash() is randomised)."""
    return zlib.crc32(str(s).encode())


# =====================================================================
# Genetic programming (symbolic regression) engine
# ---------------------------------------------------------------------
# Expression trees are nested tuples:  ('x',) | ('c',) | (op, left[, right])
# with binary ops +,-,*,/ (protected) and unary ops neg, sq, sqrt(|.|),
# nexp=exp(-|.|), log=log1p(|.|), squash=logistic.  Each tree is scored as
# y = a*f(x)+b with (a,b) by weighted least squares; fitness is a BIC so a
# more complex tree must earn its extra nodes.  Trees are bounded (~25 nodes).
# =====================================================================

_GP_BINOPS = ("+", "-", "*", "/")
_GP_UNOPS = ("neg", "sq", "sqrt", "nexp", "log", "squash")
_GP_MAX_NODES = 25


def _gp_eval(tree, x):
    """Evaluate a GP tree at ``x`` (vectorised, clip-protected)."""
    with np.errstate(all="ignore"):
        if tree[0] == "x":
            return np.asarray(x, dtype=float)
        if tree[0] == "c":
            return np.ones_like(np.asarray(x, dtype=float))
        op = tree[0]
        if op in _GP_BINOPS:
            a = _gp_eval(tree[1], x)
            b = _gp_eval(tree[2], x)
            if op == "+":
                return a + b
            if op == "-":
                return a - b
            if op == "*":
                return a * b
            return a / np.where(np.abs(b) < 1e-6,
                                np.sign(b) * 1e-6 if np.any(b) else 1e-6, b)
        a = _gp_eval(tree[1], x)
        if op == "neg":
            return -a
        if op == "sq":
            return a * a
        if op == "sqrt":
            return np.sqrt(np.abs(a))
        if op == "nexp":
            return np.exp(-np.abs(np.clip(a, -50.0, 50.0)))
        if op == "log":
            return np.log1p(np.abs(a))
        return 1.0 / (1.0 + np.exp(-np.clip(a, -50.0, 50.0)))  # squash


def _gp_nodes(tree) -> int:
    if tree[0] in ("x", "c"):
        return 1
    return 1 + sum(_gp_nodes(t) for t in tree[1:])


def _gp_random(rng, depth, full):
    if depth <= 0 or (not full and rng.random() < 0.35):
        return ("x",) if rng.random() < 0.7 else ("c",)
    if rng.random() < 0.6 or depth == 1:
        op = rng.choice(_GP_BINOPS)
        return (op, _gp_random(rng, depth - 1, full),
                _gp_random(rng, depth - 1, full))
    return (rng.choice(_GP_UNOPS), _gp_random(rng, depth - 1, full))


def _gp_subtrees(tree):
    yield tree
    for t in tree[1:]:
        if isinstance(t, tuple):
            yield from _gp_subtrees(t)


def _gp_replace(tree, target_id, new, _cid=None):
    if _cid is None:
        _cid = [0]
    if _cid[0] == target_id:
        return new
    if tree[0] in ("x", "c"):
        _cid[0] += 1
        return tree
    _cid[0] += 1
    return (tree[0],) + tuple(
        _gp_replace(t, target_id, new, _cid) if isinstance(t, tuple) else t
        for t in tree[1:])


def _gp_fitness(tree, x, y, w):
    """Return (bic, (a, b)) for y ~ a*f(x)+b fitted by weighted OLS."""
    try:
        f = np.asarray(_gp_eval(tree, x), dtype=float)
        m = np.isfinite(f)
        if m.sum() < 3 or np.std(f[m]) <= 1e-12:
            return np.inf, (0.0, 0.0)
        ff, yy, ww = f[m], y[m], w[m]
        A = np.vstack([ff, np.ones_like(ff)]).T * np.sqrt(ww)[:, None]
        yw = yy * np.sqrt(ww)
        (a, b), _, _, _ = np.linalg.lstsq(A, yw, rcond=None)
        resid = yy - (a * ff + b)
        rss = float(np.sum(ww * resid * resid))
        n = float(np.sum(ww > 0))
        if not np.isfinite(rss) or rss <= 0:
            return np.inf, (0.0, 0.0)
        bic = n * np.log(rss / max(n, 1e-9)) \
            + _gp_nodes(tree) * np.log(max(n, 2.0))
        return float(bic), (float(a), float(b))
    except Exception:
        return np.inf, (0.0, 0.0)


def _gp_to_str(tree) -> str:
    if tree[0] == "x":
        return "x"
    if tree[0] == "c":
        return "1"
    if tree[0] in _GP_BINOPS:
        return f"({_gp_to_str(tree[1])} {tree[0]} {_gp_to_str(tree[2])})"
    return f"{tree[0]}({_gp_to_str(tree[1])})"


def _gp_to_latex(tree) -> str:
    """Render a discovered tree as infix LaTeX math."""
    if tree[0] == "x":
        return "x"
    if tree[0] == "c":
        return "1"
    if tree[0] in _GP_BINOPS:
        a, b = _gp_to_latex(tree[1]), _gp_to_latex(tree[2])
        if tree[0] == "+":
            return f"{a} + {b}"
        if tree[0] == "-":
            return f"{a} - {b}"
        if tree[0] == "*":
            return f"{a} \\cdot {b}"
        return f"\\frac{{{a}}}{{{b}}}"
    a = _gp_to_latex(tree[1])
    if tree[0] == "neg":
        return f"-{a}"
    if tree[0] == "sq":
        return f"{a}^{{2}}"
    if tree[0] == "sqrt":
        return f"\\sqrt{{|{a}|}}"
    if tree[0] == "nexp":
        return f"e^{{-|{a}|}}"
    if tree[0] == "log":
        return f"\\log(1 + |{a}|)"
    return f"\\sigma({a})"


def gp_discover_form(x, y, w=None, seed_trees=(("x",), ("c",)),
                     pop=16, gens=20, seed=0, budget_s=60.0, max_depth=4):
    """Bounded symbolic regression.  Returns ``(tree, (a, b), bic, form_str)``.

    Seeds guarantee the theory shapes are in the gene pool; elitism keeps the
    best, so the returned form never loses to the seeds.  Any failure (or
    insufficient data) returns ``(None, (0,0), inf, reason)`` and the caller
    falls back to the analytic form.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[m], y[m], w[m]
    if len(x) < 8 or np.std(y) <= 0:
        return None, (0.0, 0.0), np.inf, "insufficient-data"
    t0 = time.time()
    rng = np.random.default_rng(int(seed))
    population = list(seed_trees)
    while len(population) < pop:
        population.append(
            _gp_random(rng, int(rng.integers(1, max(2, max_depth))),
                       bool(rng.integers(0, 2))))
    scored = [(_gp_fitness(t, x, y, w), t) for t in population]
    scored.sort(key=lambda s: s[0][0])
    elite = [scored[0][1], scored[1][1]] if len(scored) > 1 else [scored[0][1]]
    for _ in range(int(gens)):
        if time.time() - t0 > float(budget_s):
            break
        children = list(elite)
        while len(children) < pop:
            r = rng.random()
            p1 = scored[int(rng.integers(0, max(2, pop // 2)))][1]
            if r < 0.55:
                p2 = scored[int(rng.integers(0, max(2, pop // 2)))][1]
                s1 = list(_gp_subtrees(p1))
                s2 = list(_gp_subtrees(p2))
                child = _gp_replace(p1, int(rng.integers(0, len(s1))),
                                    s2[int(rng.integers(0, len(s2)))])
            elif r < 0.8:
                s1 = list(_gp_subtrees(p1))
                child = _gp_replace(p1, int(rng.integers(0, len(s1))),
                                    _gp_random(rng, 2, False))
            else:
                child = _gp_random(rng, 3, False)
            if _gp_nodes(child) > _GP_MAX_NODES:
                child = p1
            children.append(child)
        scored = [(_gp_fitness(t, x, y, w), t) for t in children]
        scored.sort(key=lambda s: s[0][0])
        elite = [scored[0][1]]
    (bic, (a, b)), best = scored[0]
    if not np.isfinite(bic):
        return None, (0.0, 0.0), np.inf, "no-finite-form"
    return best, (a, b), bic, f"{a:.4g}*({_gp_to_str(best)})+{b:.4g}"


# =====================================================================
# Data helpers
# =====================================================================

DEFAULT_LIMIT_COLS = ("speed_limit", "maxspeed_kmh")


def _numeric(frame, col, default=0.0):
    if col in frame.columns:
        return pd.to_numeric(frame[col], errors="coerce")
    return pd.Series(default, index=frame.index, dtype=float)


def _sd_limit(frame, limit_cols=DEFAULT_LIMIT_COLS, default=50.0):
    """Posted speed limit per row: first present limit column, else ``default``."""
    for col in limit_cols:
        if col in frame.columns:
            v = pd.to_numeric(frame[col], errors="coerce")
            if v.notna().any():
                return v.fillna(v.median()).to_numpy(float)
    return np.full(len(frame), float(default))


def _panel(df, q_col, v_col, cls_col, y_col=None):
    d = pd.DataFrame({
        "cls": (df[cls_col].fillna("unclassified").astype(str)
                if cls_col in df.columns else "unclassified"),
        "q": _numeric(df, q_col).fillna(0.0),
        "v": _numeric(df, v_col, default=np.nan),
    })
    if y_col is not None:
        d["y"] = _numeric(df, y_col).fillna(0.0)
    return d[np.isfinite(d["v"].to_numpy()) & (d["v"].to_numpy() > 0)].copy()


# =====================================================================
# Analytic curves (theory path)
# =====================================================================

def fit_speed_density(df, q_col="vehicle_count",
                      v_col="mean_vehicle_speed_kmh", cls_col="highway",
                      min_rows=200, penetration=0.05, bin_hours=0.25,
                      kj_prior_real=120.0, kj_mode="theory",
                      shrink_rows=2000.0):
    """Greenshields ``v = vf*(1 - k/kj)`` per highway class.

    Jam density ``kj`` is physics (standstill spacing ~6-8 m -> ~120-165
    veh/km/lane), not a statistical parameter, so ``kj_mode="theory"`` (the
    default) fixes it from geometry and estimates only the densely-observed
    free-flow speed ``vf``.  ``k`` is on the probe *count* scale
    (``k_real = k / (bin_hours * penetration)``), consistent between fit and
    predict.

    Returns ``{class: (vf, kj)}`` plus ``_global``, ``_diag`` and the theory
    anchors (``_theory_kj_cnt`` etc.) for auditing.
    """
    d = _panel(df, q_col, v_col, cls_col)
    scales = 1.0 / max(float(bin_hours) * max(float(penetration), 1e-9), 1e-12)
    kj_thy = float(kj_prior_real) / scales
    mode = str(kj_mode or "theory").lower()

    def _one(g):
        if len(g) < min_rows:
            return None
        k = (g["q"] / g["v"]).to_numpy(float)
        v = g["v"].to_numpy(float)
        cap = float(np.quantile(k, 0.99)) if len(k) else 0.0
        keep = k <= max(cap, 1e-9)
        k, v = k[keep], v[keep]
        if len(k) < min_rows:
            return None
        klo = k <= np.quantile(k, 0.5)
        kk, vv = (k[klo], v[klo]) if klo.sum() >= min_rows else (k, v)
        A = np.vstack([np.ones_like(kk), kk]).T
        (a, b), _, _, _ = np.linalg.lstsq(A, vv, rcond=None)
        vf = float(np.clip(a, 5.0, 130.0))
        n_near = float((k >= 0.5 * kj_thy).sum())
        kj_ols = float(-vf / b) if b < -1e-9 else float("nan")
        if mode == "ols":
            kj = kj_ols if np.isfinite(kj_ols) else kj_thy
            kj = float(max(kj, 0.9 * kj_thy))
        elif mode == "eb":
            w = n_near / (n_near + max(float(shrink_rows), 1.0))
            kj = (w * kj_ols + (1.0 - w) * kj_thy) \
                if np.isfinite(kj_ols) else kj_thy
            kj = float(max(kj, 0.9 * kj_thy))
        else:
            kj = kj_thy
        diag = {"vf": round(vf, 2), "kj": round(float(kj), 3), "kj_mode": mode,
                "kj_ols": (round(float(kj_ols), 3)
                           if np.isfinite(kj_ols) else None),
                "n_near_jam": int(n_near), "n_rows": int(len(g)),
                "vf_fit_rows": int(len(kk))}
        return vf, kj, diag

    out, diag = {}, {}
    for cls, g in list(d.groupby("cls")) + [("_global", d)]:
        r = _one(g)
        if r is not None:
            out[str(cls)] = (r[0], r[1])
            diag[str(cls)] = r[2]
    out["_diag"] = diag
    out["_theory_kj_cnt"] = kj_thy
    out["_theory_kj_real"] = float(kj_prior_real)
    out["_density_scale"] = scales
    out["_kj_mode"] = mode
    return out


def fit_propensity_lookup(df, target, q_col="vehicle_count",
                          v_col="mean_vehicle_speed_kmh", cls_col="highway",
                          limit_cols=DEFAULT_LIMIT_COLS):
    """Empirical speeding rate per (class, exceedance bin), smoothed.

    ``rate = sum(Y) / sum(q)`` per cell over the ``SD_EDGES`` bins of
    ``e = v - limit``; this is the behavioural link density -> speed ->
    exceedance -> rate.  Fit on train windows only.  Returns
    ``{class: rate_array}`` (length ``len(SD_EDGES)-1``) plus ``_global``.
    """
    d = _panel(df, q_col, v_col, cls_col, y_col=target)
    d["lim"] = _sd_limit(df, limit_cols)
    d["e"] = d["v"] - d["lim"]
    d["bin"] = np.clip(np.digitize(d["e"].to_numpy(float), SD_EDGES) - 1,
                       0, len(SD_EDGES) - 2)
    nb = len(SD_EDGES) - 1
    out = {}
    for cls, g in list(d.groupby("cls")) + [("_global", d)]:
        agg = g.groupby("bin").agg(ys=("y", "sum"), qs=("q", "sum"))
        rate = (agg["ys"] / agg["qs"].clip(lower=1e-9)).reindex(
            range(nb), fill_value=np.nan)
        rate = rate.fillna(rate.mean()).fillna(0.0)
        rate = rate.rolling(3, center=True, min_periods=1).mean()
        out[str(cls)] = rate.to_numpy(float)
    return out


# =====================================================================
# GP form discovery over the SD curves
# =====================================================================

def fit_sd_forms(df, target, sd_curves, q_col="vehicle_count",
                 v_col="mean_vehicle_speed_kmh", cls_col="highway",
                 limit_cols=DEFAULT_LIMIT_COLS, budget_s=240.0, seed=0):
    """Per-class GP form discovery for the SD loop (bounded, seeded).

    Speed shape is discovered on quantile-binned ``(k, v)``; propensity shape
    on binned ``log``-rates over exceedance.  Seeds: linear + constant for
    speed (linear-in-k = Greenshields), linear + squash for propensity.  A
    timeout or failure leaves the entry ``None`` so the analytic path stands.

    Returns ``{cls: {"speed": (tree, (a,b), form_str, bic) | None,
                     "prop":  ... | None}}``.
    """
    t0 = time.time()
    d = _panel(df, q_col, v_col, cls_col, y_col=target)
    d["lim"] = _sd_limit(df, limit_cols)
    d["k"] = d["q"] / d["v"].clip(lower=1.0)
    d["e"] = d["v"] - d["lim"]
    out = {}
    for cls, g in d.groupby("cls"):
        if time.time() - t0 > float(budget_s):
            break
        entry = {"speed": None, "prop": None}
        try:
            qb = pd.qcut(g["k"], 12, duplicates="drop")
            b = g.groupby(qb, observed=True).agg(
                k=("k", "mean"), v=("v", "mean"), n=("v", "size")).dropna()
            if len(b) >= 8:
                tree, ab, bic, fs = gp_discover_form(
                    b["k"].to_numpy(), b["v"].to_numpy(),
                    w=b["n"].to_numpy(),
                    seed_trees=(("x",), ("c",), ("-", ("x",), ("c",))),
                    pop=12, gens=12, seed=seed + (_crc(str(cls)) % 1000),
                    budget_s=15.0)
                if tree is not None:
                    entry["speed"] = (tree, ab, fs, round(float(bic), 1))
            eb = np.clip(np.digitize(g["e"].to_numpy(float), SD_EDGES) - 1,
                         0, len(SD_EDGES) - 2)
            gg = g.assign(_eb=eb).groupby("_eb").agg(
                ys=("y", "sum"), qs=("q", "sum"), e=("e", "mean"))
            gg = gg[gg["qs"] > 0].copy()
            gg["lr"] = np.log((gg["ys"] / gg["qs"]).clip(lower=1e-9))
            if len(gg) >= 8 and np.std(gg["lr"].to_numpy()) > 1e-9:
                tree, ab, bic, fs = gp_discover_form(
                    gg["e"].to_numpy(), gg["lr"].to_numpy(),
                    w=gg["qs"].to_numpy(),
                    seed_trees=(("x",), ("c",), ("squash", ("x",))),
                    pop=12, gens=12,
                    seed=seed + 54321 + (_crc(str(cls)) % 1000),
                    budget_s=15.0)
                if tree is not None:
                    entry["prop"] = (tree, ab, fs, round(float(bic), 1))
        except Exception:
            pass
        out[str(cls)] = entry
    return out


# =====================================================================
# Forward simulation (shared by both paths)
# =====================================================================

def sd_regime(q, cls_arr, sd_curves):
    """Share of rows in each band of ``rho = k/kj``.

    If the congestion-active share (0.5-0.9) is ~0, the structurally-identified
    twin *cannot* produce congestion suppression in this data, so any negative
    game lift must be identified by the regression instead.  That is a finding
    about the data, not a bug in the twin.
    """
    q = np.maximum(np.asarray(q, dtype=float), 0.0)
    cls_arr = np.asarray(cls_arr, dtype=str)
    glob = sd_curves.get("_global", (50.0, 1.5))
    vf = np.array([sd_curves.get(c, glob)[0] for c in cls_arr], float)
    kj = np.array([sd_curves.get(c, glob)[1] for c in cls_arr], float)
    with np.errstate(all="ignore"):
        k = q / np.clip(vf, 1.0, None)
        rho = k / np.clip(kj, 1e-9, None)
    rho = np.clip(np.where(np.isfinite(rho), rho, 0.0), 0.0, 2.0)
    n = max(len(rho), 1)
    return {
        "free_flow_rho_lt_.2": round(float((rho < 0.2).sum()) / n, 4),
        "transition_.2_.5": round(float(((rho >= 0.2) & (rho < 0.5)).sum()) / n, 4),
        "congestion_active_.5_.9": round(
            float(((rho >= 0.5) & (rho < 0.9)).sum()) / n, 4),
        "jam_ge_.9": round(float((rho >= 0.9).sum()) / n, 4),
        "rho_median": round(float(np.median(rho)), 4),
        "rho_p95": round(float(np.percentile(rho, 95)), 4),
        "n_rows": int(len(rho)),
    }


def sd_predict(q, cls_arr, lim_arr, sd_curves, prop_lookup, sd_forms=None):
    """Forward-simulate ``q -> k=q/v -> v -> e -> rate -> counts``.

    Parameter path: Greenshields ``v(vf,kj)`` + binned rate table.  GP path
    (per class, when discovery found a form): ``v = a*f(k)+b`` and
    ``log-rate = a*g(e)+b`` through the same fixed point.  Fixed point capped
    at 3 rounds and everything clipped, so a wild form degrades to bounds.

    Returns ``(counts, v, k)``.
    """
    q = np.maximum(np.asarray(q, dtype=float), 0.0)
    cls_arr = np.asarray(cls_arr, dtype=str)
    glob = sd_curves.get("_global", (50.0, np.inf))
    vf = np.array([sd_curves.get(c, glob)[0] for c in cls_arr], float)
    kj = np.array([sd_curves.get(c, glob)[1] for c in cls_arr], float)
    forms = sd_forms or {}
    has_speed = np.array([bool((forms.get(c) or {}).get("speed"))
                          for c in cls_arr])
    v = vf.copy()
    for _ in range(3):
        k = q / np.clip(v, 1.0, None)
        v_par = np.where(np.isfinite(kj), vf * (1.0 - k / kj), vf)
        if has_speed.any():
            v_gp = v.copy()
            for c in np.unique(cls_arr[has_speed]):
                m = (cls_arr == c) & has_speed
                tree, (a, b) = forms[c]["speed"][0], forms[c]["speed"][1]
                with np.errstate(all="ignore"):
                    fv = a * np.asarray(_gp_eval(tree, k[m]), dtype=float) + b
                fv = np.clip(np.where(np.isfinite(fv), fv, vf[m]),
                             1.0, np.maximum(vf[m], 1.0) * 1.5)
                v_gp[m] = fv
            v = np.where(has_speed, v_gp, v_par)
        else:
            v = v_par
        v = np.clip(v, 1.0, np.maximum(vf, 1.0) * 1.5)
    e = v - np.asarray(lim_arr, dtype=float)
    bi = np.clip(np.digitize(e, SD_EDGES) - 1, 0, len(SD_EDGES) - 2)
    nb = len(SD_EDGES) - 1
    g_rates = prop_lookup.get("_global", np.zeros(nb))
    rates = np.array([np.asarray(prop_lookup.get(c, g_rates), dtype=float)[b]
                      for c, b in zip(cls_arr, bi)], float)
    has_prop = np.array([bool((forms.get(c) or {}).get("prop"))
                         for c in cls_arr])
    if has_prop.any():
        r_gp = rates.copy()
        for c in np.unique(cls_arr[has_prop]):
            m = (cls_arr == c) & has_prop
            tree, (a, b) = forms[c]["prop"][0], forms[c]["prop"][1]
            with np.errstate(all="ignore"):
                lr = a * np.asarray(_gp_eval(tree, e[m]), dtype=float) + b
            r_gp[m] = np.exp(np.clip(np.where(np.isfinite(lr), lr, -20.0),
                                     -20.0, 1.0))
        rates = np.where(has_prop, r_gp, rates)
    counts = q * np.clip(rates, 0.0, 2.0)
    return counts, v, q / np.clip(v, 1.0, None)


# =====================================================================
# Convenience wrapper
# =====================================================================

class SystemDynamicsModel:
    """Fit and forward-simulate the speeding stock-flow twin.

    Parameters
    ----------
    target : str
        Column holding the observed speeding count (the rate numerator).
    q_col / v_col / cls_col : str
        Flow, speed and highway-class columns in the panel.
    limit_cols : tuple[str, ...]
        Candidate posted-speed-limit columns (first non-null wins).
    form_mode : {"theory", "search"}
        ``theory`` uses the analytic curves; ``search`` runs the GP form
        discovery (seeded with theory, so it can only tie-or-beat it).
    gp_budget_s : float
        Wall-time budget for the whole GP discovery pass.
    """

    def __init__(self, target, q_col="vehicle_count",
                 v_col="mean_vehicle_speed_kmh", cls_col="highway",
                 limit_cols=DEFAULT_LIMIT_COLS, form_mode="theory",
                 gp_budget_s=240.0, seed=0, **speed_density_kw):
        self.target = target
        self.q_col = q_col
        self.v_col = v_col
        self.cls_col = cls_col
        self.limit_cols = tuple(limit_cols)
        self.form_mode = str(form_mode or "theory").lower()
        self.gp_budget_s = float(gp_budget_s)
        self.seed = int(seed)
        self._sd_kw = speed_density_kw
        self.sd_curves = None
        self.prop_lookup = None
        self.sd_forms = None

    def fit(self, df):
        """Fit the analytic curves (and GP forms when ``form_mode="search"``)."""
        self.sd_curves = fit_speed_density(
            df, q_col=self.q_col, v_col=self.v_col, cls_col=self.cls_col,
            **self._sd_kw)
        self.prop_lookup = fit_propensity_lookup(
            df, self.target, q_col=self.q_col, v_col=self.v_col,
            cls_col=self.cls_col, limit_cols=self.limit_cols)
        if self.form_mode == "search":
            self.sd_forms = fit_sd_forms(
                df, self.target, self.sd_curves, q_col=self.q_col,
                v_col=self.v_col, cls_col=self.cls_col,
                limit_cols=self.limit_cols, budget_s=self.gp_budget_s,
                seed=self.seed)
        else:
            self.sd_forms = None
        return self

    def simulate(self, q, cls_arr, limit_arr=None, df_for_limit=None):
        """Forward-simulate predicted counts.  Returns ``(counts, v, k)``."""
        if self.sd_curves is None:
            raise RuntimeError("call fit() first")
        if limit_arr is None:
            if df_for_limit is None:
                limit_arr = np.full(len(np.asarray(q)), 50.0)
            else:
                limit_arr = _sd_limit(df_for_limit, self.limit_cols)
        return sd_predict(np.asarray(q, dtype=float), np.asarray(cls_arr),
                          np.asarray(limit_arr, dtype=float),
                          self.sd_curves, self.prop_lookup, self.sd_forms)

    def regimes(self, q, cls_arr):
        """Density-regime shares for the given flow/class arrays."""
        return sd_regime(q, cls_arr, self.sd_curves)

    def summary(self):
        """Compact audit dict: curve params and any discovered forms."""
        out = {"form_mode": self.form_mode,
               "kj_mode": (self.sd_curves or {}).get("_kj_mode"),
               "curves": {k: v for k, v in (self.sd_curves or {}).items()
                          if not str(k).startswith("_")}}
        if self.sd_forms:
            out["forms"] = {
                cls: {role: (None if entry is None
                             else {"form": entry[2], "bic": entry[3]})
                      for role, entry in (payload or {}).items()}
                for cls, payload in self.sd_forms.items()}
        return out


def fit_system_dynamics(df, target, **kwargs) -> SystemDynamicsModel:
    """One-call convenience: build, fit and return a ``SystemDynamicsModel``."""
    return SystemDynamicsModel(target, **kwargs).fit(df)
