"""Compare draw-based (Halton) vs draw-free (GHQ, AGHQ) integration."""
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from metacountregressor import GA_CMF_AADT_JAX as G


def sim(n, seed=0, sig=0.4):
    rng = np.random.default_rng(seed)
    aadt = rng.uniform(1000, 20000, n)
    x, z = rng.normal(size=(2, n))
    mu = np.exp(-8.0 + (0.3 + sig * rng.normal(size=n)) * x
                + (0.9 + 0.2 * z) * np.log(aadt))
    return pd.DataFrame({"FREQ": rng.poisson(mu), "AADT": aadt, "X": x, "Z": z})


n = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
df = sim(n)
rows = []
for name, kw in [("halton R=100", dict(integration="halton", R=100)),
                 ("halton R=500", dict(integration="halton", R=500)),
                 ("ghq 7", dict(integration="ghq", ghq_points=7)),
                 ("ghq 15", dict(integration="ghq", ghq_points=15)),
                 ("aghq 5", dict(integration="aghq", ghq_points=5))]:
    t = time.time()
    res, y, A, B, L, draws, _, off = G.fit_final_model(
        df, ["X"], ["Z"], [True], [False], model="poisson", **kw)
    dt = time.time() - t
    p = np.asarray(res.params)
    mb = 0 if draws is None else draws.size * draws.dtype.itemsize / 1e6
    rows.append({"method": name, "seconds": round(dt, 1), "draw_MB": round(mb, 1),
                 "alpha_X": round(p[1], 4), "sigma_X": round(abs(p[2]), 4),
                 "beta0": round(p[3], 4), "beta_Z": round(p[4], 4)})
    print(rows[-1], flush=True)
print(pd.DataFrame(rows).to_string(index=False))
