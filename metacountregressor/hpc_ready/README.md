# MetaCount — HPC-ready Experiment Scripts

Drop this entire `hpc_ready/` folder onto your Linux HPC system alongside the
`metacountregressor/` package directory.  No pip install is required — the
scripts locate the package automatically via relative path.

```
your_cluster_workdir/
├── metacountregressor/          ← repo checkout (the package root)
│   ├── main_hpc.py
│   ├── main_hpc_lc_patch.py
│   ├── experiment_package.py
│   └── ...
└── hpc_ready/                   ← copy this folder
    ├── run_experiment.py
    ├── run_experiment.sh
    ├── requirements.txt
    ├── data/                    ← put your CSV data files here
    └── README.md
```

---

## 1. Environment setup

```bash
# Option A — conda
conda create -n metacount python=3.11
conda activate metacount
pip install -r requirements.txt

# Option B — venv
python -m venv ../.venv
source ../.venv/bin/activate
pip install -r requirements.txt
```

Edit the environment-activation block near the top of `run_experiment.sh`
to match your cluster's module/conda setup.

---

## 2. Quick smoke-test (synthetic data, no CSV needed)

```bash
# Interactive (no SLURM):
bash run_experiment.sh synthetic_both

# Via SLURM:
sbatch run_experiment.sh synthetic_both
```

---

## 3. Latent-class count model (NB2) on real data

```bash
# Manual fit — 1-class and 2-class NB2:
sbatch run_experiment.sh count \
    --data data/crashes.csv \
    --id_col site_id  --y_col crash_count \
    --variables "aadt,lanes,speed,shoulder,hurst" \
    --membership_vars "entry_ramp,exit_ramp" \
    --n_classes 2

# SA specification search — up to 3 classes:
sbatch run_experiment.sh search_count \
    --data data/crashes.csv \
    --id_col site_id  --y_col crash_count \
    --max_classes 3  --search_iter 5000
```

---

## 4. Latent-class Tobit model on real data

Use when your outcome is a *rate* (crash rate per 100M VMT) that is
left-censored at zero.

```bash
# Manual fit — 1-class and 2-class LC-Tobit:
sbatch run_experiment.sh tobit \
    --data data/crash_rates.csv \
    --id_col site_id  --y_col rate_per_100mveh \
    --variables "h085,aadt_per_lane,citybound,pct_heavy" \
    --membership_vars "entry_ramp,exit_ramp" \
    --n_classes 2

# SA specification search — up to 2 classes:
sbatch run_experiment.sh search_tobit \
    --data data/crash_rates.csv \
    --id_col site_id  --y_col rate_per_100mveh \
    --max_classes 2  --search_iter 3000
```

---

## 5. Run both families in a single job

```bash
sbatch run_experiment.sh both \
    --data data/crashes.csv \
    --id_col site_id  --y_col crashes \
    --variables "aadt,lanes,speed" \
    --n_classes 2
```

---

## 6. Output structure

```
results/
└── <timestamp>_<tag>/
    ├── lc_count/
    │   ├── lc1_nb_stats.json          BIC / LL / k / n
    │   ├── lc1_nb_params.csv          raw parameter vector
    │   ├── lc1_nb_predictions.csv     per-observation predictions
    │   ├── lc2_nb_stats.json
    │   ├── lc2_nb_params.csv
    │   ├── lc2_nb_predictions.csv
    │   └── lc_count_comparison.csv    BIC comparison table
    └── lc_tobit/
        ├── lc1_tobit_stats.json
        ├── lc2_tobit_stats.json
        └── lc_tobit_comparison.csv
```

---

## 7. SLURM resource tuning

Edit the `#SBATCH` directives at the top of `run_experiment.sh`:

| Directive              | Recommended starting point                |
|------------------------|-------------------------------------------|
| `--time`               | `04:00:00` for manual; `12:00:00` search  |
| `--cpus-per-task`      | 8 (JAX uses intra-op threading)           |
| `--mem`                | 32G for N<2000; 64G for larger datasets   |
| `--partition`          | match your cluster's CPU/GPU partition    |

JAX automatically uses all available cores.  Set `XLA_FLAGS` in the script
to cap threading if you share a node.

---

## 8. Key arguments reference

| Argument            | Default      | Description                                  |
|---------------------|--------------|----------------------------------------------|
| `--experiment`      | `both`       | `lc_count`, `lc_tobit`, `both`               |
| `--mode`            | `manual`     | `manual` (fits 1..n_classes) or `search`     |
| `--data`            | *(synthetic)*| Path to CSV relative to hpc_ready/           |
| `--id_col`          | `id`         | Panel entity identifier column               |
| `--y_col`           | `y`          | Outcome column                               |
| `--offset_col`      | *(none)*     | Log-exposure offset (count models only)      |
| `--variables`       | all cols     | Comma-separated outcome predictor names      |
| `--membership_vars` | *(none)*     | Comma-separated class-membership covariates  |
| `--n_classes`       | `2`          | Fit models up to this many classes           |
| `--max_classes`     | `3`          | Upper bound for search mode                  |
| `--n_draws`         | `200`        | Halton draws for mixed / random-param models |
| `--search_iter`     | `3000`       | SA iterations in search mode                 |
| `--output_dir`      | `results`    | Output root directory                        |
| `--seed`            | `42`         | RNG seed for reproducibility                 |
