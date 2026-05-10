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

import subprocess
import sys
import time
from pathlib import Path

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
    sys.executable, str(SCRIPT),
    "--input",             str(DATA),
    "--output-dir",        str(OUT),
    "--search-iter",       "120",
    "--family",            "nb",
    "--candidate-profile", "expanded",
    "--allow-nonmonotonic-fallback",
])

print("=" * 60)
print("  Step 2/2 — Rendering presentation ...")
print("=" * 60)
run(["quarto", "render", str(QMD)])

html = QMD.with_suffix(".html")
print(f"\nDone in {time.time()-t0:.0f}s")
print(f"\nOpen:  {html}")
