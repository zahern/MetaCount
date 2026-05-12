#!/usr/bin/env python3
"""
Run this file. That's it.

    python build_washington_presentation.py

It will:
  1. Run the Washington CMF experiment (120 iterations, NB2 model)
  2. Generate three interactive HTML files in results/ex16_3_cmf/
  3. Render the Quarto presentation
  4. Print the path to the finished presentation
"""

import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

# Suppress statsmodels NB2 numerical warnings — they are benign intermediate
# states during optimisation (overflow/log(0) in gradient steps that recover).
warnings.filterwarnings("ignore")
# Pass suppression flags into the subprocess environment so the XLA/JAX
# C++ runtime message ("Empty bitcode string for eigen") is silenced there too.
os.environ["PYTHONWARNINGS"]     = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"    # suppress XLA C++ runtime logs
os.environ["JAX_PLATFORMS"]      = "cpu"    # explicit platform avoids startup noise

ROOT   = Path(__file__).resolve().parent
SCRIPT = ROOT / "scripts" / "generate_washington_hierarchical_cmf_assets.py"
QMD    = ROOT / "scripts" / "washington_cmf_presentation.qmd"
DATA   = ROOT / "data" / "Ex-16-3.csv"
OUT    = ROOT / "results" / "ex16_3_cmf"

def run(cmd, **kw):
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n")
    r = subprocess.run(cmd, **kw)
    if r.returncode != 0:
        print(f"\nERROR: command failed (exit {r.returncode})", file=sys.stderr)
        sys.exit(r.returncode)

t0 = time.time()
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  Step 1/2 — Running experiment ...")
print("=" * 60)
run([
    sys.executable, "-W", "ignore", str(SCRIPT),
    "--input",             str(DATA),
    "--output-dir",        str(OUT),
    "--search-iter",       "600",
    "--families",          "both",          # search NB2 + Poisson, pick best BIC
    "--family",            "nb",            # final refit family
    "--candidate-profile", "expanded",
    "--max-upper-terms",   "6",
    "--max-lower-terms",   "4",
    "--allow-nonmonotonic-fallback",
])

print("=" * 60)
print("  Step 2/2 — Rendering presentation ...")
print("=" * 60)
run(["quarto", "render", str(QMD)])

html = QMD.with_suffix(".html")
pptx = QMD.with_suffix(".pptx")
print(f"\nDone in {time.time()-t0:.0f}s")
print(f"\nHTML presentation:  {html}")
if pptx.exists():
    print(f"PPTX presentation:  {pptx}")

# List all standalone interactive HTML files — open any of these fullscreen in a browser
interactive = [
    ("CMF Dashboard (live controls)",      OUT / "hierarchical_cmf_dashboard.html"),
    ("Search Convergence (BIC + RMSE)",    OUT / "search_convergence.html"),
    ("Model Comparison (side-by-side)",    OUT / "model_comparison.html"),
    ("AADT Obs vs Predicted",              OUT / "aadt_obs_pred.html"),
]
print("\nInteractive HTML files (open directly in browser for fullscreen):")
for label, p in interactive:
    if p.exists():
        print(f"  {label:<38}  {p}")
