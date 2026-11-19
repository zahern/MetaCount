"""
zeke_minimize.py
========================================================================
Tiny numpy-only optimisation helpers used by the pavement modules.

Provides drop-in replacements for the two ``scipy.optimize`` routines the
pavement code needs:

    minimize_scalar(f, bounds=(lo, hi), method="bounded")  -> OptimizeResult
    minimize(f, x0, method="Nelder-Mead")                  -> OptimizeResult

If SciPy is available it is used (faster / more robust); otherwise a
self-contained implementation is used so the modules run on a bare
``numpy`` install. This keeps the new pavement layers embeddable into
``metacountregressor`` environments that may not ship SciPy.
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.optimize import minimize as _sp_minimize
    from scipy.optimize import minimize_scalar as _sp_minimize_scalar
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


class OptimizeResult:
    def __init__(self, x, fun):
        self.x = np.asarray(x, float)
        self.fun = float(fun)


# ---------------------------------------------------------------------------
# Stat helpers (numpy-only fallbacks for scipy.special)
# ---------------------------------------------------------------------------
def softmax(z, axis=-1):
    z = np.asarray(z, float)
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def expit(x):
    x = np.asarray(x, float)
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# Bounded scalar minimisation (golden section)
# ---------------------------------------------------------------------------
def minimize_scalar(fun, bounds=None, method="bounded", xatol=1e-8):
    if HAS_SCIPY:
        # scipy expects tolerances inside `options`, not as a keyword
        return _sp_minimize_scalar(fun, bounds=bounds, method=method,
                                   options={"xatol": xatol})
    lo, hi = bounds
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = float(lo), float(hi)
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc, fd = fun(c), fun(d)
    for _ in range(100):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = fun(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = fun(d)
        if abs(b - a) < xatol:
            break
    x = 0.5 * (a + b)
    return OptimizeResult(x, fun(x))


# ---------------------------------------------------------------------------
# Nelder-Mead for unconstrained R^n (used by hazard + Markov MLE)
# ---------------------------------------------------------------------------
def minimize(fun, x0, method="Nelder-Mead", options=None, args=()):
    if HAS_SCIPY:
        return _sp_minimize(fun, x0, method=method, options=options, args=args)
    options = options or {}
    maxiter = int(options.get("maxiter", 2000))
    xatol = float(options.get("xatol", 1e-6))
    fatol = float(options.get("fatol", 1e-6))
    rho, chi, psi, sigma = 1.0, 0.5, 2.0, 0.5
    x0 = np.asarray(x0, float)
    n = x0.size
    # initial simplex
    simp = [x0.copy()]
    for i in range(n):
        v = x0.copy(); v[i] += 0.1 * (abs(x0[i]) + 0.1)
        simp.append(v)
    fvals = [fun(s, *args) for s in simp]

    for _ in range(maxiter):
        order = np.argsort(fvals)
        if (fvals[order[0]] - fvals[order[-1]]) < fatol:
            break
        if np.max(np.abs(simp[order[0]] - simp[order[-1]])) < xatol:
            break
        best = simp[order[0]]
        # centroid of all but worst
        centroid = np.mean([simp[i] for i in order[:-1]], axis=0)
        # reflection
        xr = centroid + rho * (centroid - simp[order[-1]])
        fr = fun(xr, *args)
        if fr < fvals[order[0]]:
            xe = centroid + psi * (xr - centroid); fe = fun(xe, *args)
            if fe < fr:
                simp[order[-1]], fvals[order[-1]] = xe, fe
            else:
                simp[order[-1]], fvals[order[-1]] = xr, fr
        elif fr < fvals[order[-2]]:
            simp[order[-1]], fvals[order[-1]] = xr, fr
        else:
            xc = centroid + chi * (simp[order[-1]] - centroid)
            fc = fun(xc, *args)
            if fc < fvals[order[-1]]:
                simp[order[-1]], fvals[order[-1]] = xc, fc
            else:
                for i in order[1:]:
                    simp[i] = simp[order[0]] + sigma * (simp[i] - simp[order[0]])
                    fvals[i] = fun(simp[i], *args)
    order = np.argsort(fvals)
    return OptimizeResult(simp[order[0]], fvals[order[0]])


if __name__ == "__main__":
    r1 = minimize_scalar(lambda x: (x - 2) ** 2, bounds=(-5, 5))
    print("scalar min at", round(float(r1.x), 4), "fun", round(r1.fun, 6))
    r2 = minimize(lambda x: (x[0] - 1) ** 2 + (x[1] + 2) ** 2,
                  [0.0, 0.0], options={"maxiter": 2000})
    print("vec min at", np.round(r2.x, 4), "fun", round(r2.fun, 6))
    print("HAS_SCIPY =", HAS_SCIPY)
