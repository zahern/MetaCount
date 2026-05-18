MetaCount benchmark re-estimation (Ex16-3 structure)

LogLik: -1221.1272
BIC: 2512.3715
AIC: 2468.2543
n_obs=220, n_params=13, optimizer_iter=109
Offset used: __BENCHMARK_OFFSET_AADT__ = log(AADT) + log(L)

Model terms:
- Nonrandom: LOWPRE, GBRPM, FRICTION
- Random means: EXPOSE, INTPM, CPM, HISNOW

Fit quality checks:
- Optimizer linesearch: PASS (failed_linesearch=False)
- Optimizer residual: PASS (error=9.040e-04)
- Objective consistency: PASS (|state.value + loglik|=2.274e-13)
- Finite coefficients: PASS
- Max |nonrandom estimate|: PASS (60.3898)
- Max random SD: WARN (13.2095)

Estimated coefficients:
| Variable | Role | Estimate | StdDev |
| --- | --- | --- | --- |
| __INTERCEPT__ | Nonrandom parameter | -8.2258 |  |
| Low-Precipitation Days per Year (precip < 1.5 in/month) | Nonrandom parameter | 60.3898 |  |
| Grade-Break Rate per Mile | Nonrandom parameter | -0.009259 |  |
| Pavement Friction, Skid Number (higher = more grip) | Nonrandom parameter | 0.105783 |  |
| Exposure Index | Random mean (normal) | -0.069410 | 11.7679 |
| Intersection Density (intersections per mile) | Random mean (normal) | 0.004960 | 13.2095 |
| Horizontal Curve Density (curves per mile) | Random mean (normal) | 0.156428 | 8.7517 |
| High-Snow Days per Year (snowfall > 1 in/day) | Random mean (normal) | 0.242622 | 7.7063 |
| NB2 Dispersion (alpha) | Dispersion | 11.0768 |  |