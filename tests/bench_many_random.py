"""All covariates random: recovery, identification and cost as q grows."""
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from metacountregressor import GA_CMF_AADT_JAX as G

n, rng = 3000, np.random.default_rng(3)
aadt = rng.uniform(1000, 20000, n)
X = rng.normal(size=(n, 6))
a_true = np.array([0.3, -0.2, 0.15]); s_up = np.array([0.3, 0.0, 0.3])
b_true = np.array([0.2, -0.1, 0.1]); s_lo = np.array([0.0, 0.03, 0.0])
eta = (-8.0 + X[:, :3] @ a_true + (X[:, :3] * s_up) @ rng.normal(size=(3, n))
       .T.diagonal().reshape(-1, 1).T.ravel().reshape(1, -1)[0:0].sum(0)
       if False else 0)
u = rng.normal(size=(n, 6))
eta = (-8.0 + X[:, :3] @ a_true + (X[:, :3] * u[:, :3]) @ s_up
       + (0.9 + X[:, 3:] @ b_true + (X[:, 3:] * u[:, 3:]) @ s_lo) * np.log(aadt))
df = pd.DataFrame({"FREQ": rng.poisson(np.exp(eta)), "AADT": aadt,
                   **{f"V{i}": X[:, i] for i in range(6)}})
up, lo = ["V0", "V1", "V2"], ["V3", "V4", "V5"]

for ru, rl in (([True, False, True], [False, True, False]),   # correct (3)
               ([True, True, True], [False, True, False]),    # + V1 (4)
               ([True, True, True], [True, True, True])):     # all random (6)
    q = sum(ru) + sum(rl)
    t = time.time()
    res, y, A, B, L, draws, _, off = G.fit_final_model(
        df, up, lo, ru, rl, model="poisson", R=200)
    dt = time.time() - t
    rep = G.identification_report(res, y, A, B, L, up, lo, ru, rl, draws,
                                  "poisson", off)
    lab = G._param_labels(up, lo, ru, rl, "poisson")
    est = {l: round(abs(float(p)) if l.startswith("sigma") else float(p), 3)
           for l, p in zip(lab, np.asarray(res.params))}
    print(f"\nq={q} random terms  {dt:.1f}s  cond={rep['condition_number']:.1e} "
          f"identified={rep['identified']} flat={rep['flat_sigmas']}")
    print({k: v for k, v in est.items() if k.startswith("sigma")})
print("\ntrue sigma: V0 .3  V2 .3  V4 .03  (others 0)")
