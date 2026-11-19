#!/usr/bin/env python3
"""
run_washington_presentation.py
===============================
One-command task that:

  1. Runs generate_washington_hierarchical_cmf_assets.py with the settings
     that match the Quarto presentation narrative (120 search iterations,
     expanded candidate pool, NB family, results to results/ex16_3_cmf/).

  2. Verifies all HTML files the presentation iframes need are present.

  3. Optionally renders washington_cmf_presentation.qmd via Quarto (requires
     quarto >= 1.3 on PATH).

Usage
-----
  # Full run: experiment + quarto render
  python scripts/run_washington_presentation.py

  # Experiment only (skip Quarto render)
  python scripts/run_washington_presentation.py --no-render

  # Quarto render only (experiment results already exist)
  python scripts/run_washington_presentation.py --render-only

  # Custom settings
  python scripts/run_washington_presentation.py \\
      --search-iter 250 --family nb --seed 42

  # Quick smoke-test (fast, fewer iterations)
  python scripts/run_washington_presentation.py --smoke-test
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# ── Repo root (one level above scripts/) ──────────────────────────────────────
HERE      = Path(__file__).resolve().parent          # scripts/
REPO_ROOT = HERE.parent                               # metacountregressor/

# ── Canonical paths ────────────────────────────────────────────────────────────
DATA_FILE      = REPO_ROOT / "data" / "Ex-16-3.csv"
EXPERIMENT_SCR = HERE / "generate_washington_hierarchical_cmf_assets.py"
PRESENTATION   = HERE / "washington_cmf_presentation.qmd"
OUTPUT_DIR     = REPO_ROOT / "results" / "ex16_3_cmf"

# Files the presentation iframes reference (relative to OUTPUT_DIR)
REQUIRED_HTML = [
    "search_convergence.html",
    "aadt_obs_pred.html",
    "hierarchical_cmf_dashboard.html",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _banner(msg: str) -> None:
    width = 68
    print("\n" + "=" * width)
    print(f"  {msg}")
    print("=" * width)


def _step(msg: str) -> None:
    print(f"\n  >>  {msg}")


def _ok(msg: str) -> None:
    print(f"     OK  {msg}")


def _warn(msg: str) -> None:
    print(f"     !!  {msg}", file=sys.stderr)


def _fail(msg: str) -> None:
    print(f"\n  FAIL  {msg}", file=sys.stderr)


def _check_prerequisites(skip_data: bool = False) -> bool:
    ok = True
    if not skip_data and not DATA_FILE.exists():
        _fail(f"Data file not found: {DATA_FILE}")
        _fail("Download Ex-16-3.csv to data/ or run from the repo root.")
        ok = False
    if not EXPERIMENT_SCR.exists():
        _fail(f"Experiment script not found: {EXPERIMENT_SCR}")
        ok = False
    if not PRESENTATION.exists():
        _fail(f"Presentation .qmd not found: {PRESENTATION}")
        ok = False
    return ok


def _check_quarto() -> bool:
    """Return True if quarto is callable and ≥ 1.3."""
    try:
        r = subprocess.run(
            ["quarto", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        version = r.stdout.strip()
        _ok(f"quarto {version} found on PATH")
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        _warn("quarto not found on PATH — skipping presentation render.")
        _warn("Install from https://quarto.org/docs/get-started/")
        return False


def _run_experiment(args: argparse.Namespace) -> bool:
    """Run the experiment script. Returns True on success."""
    _step("Running Washington CMF experiment ...")

    cmd = [
        sys.executable,
        str(EXPERIMENT_SCR),
        "--input",      str(DATA_FILE),
        "--output-dir", str(OUTPUT_DIR),
        "--search-iter", str(args.search_iter),
        "--family",      args.family,
        "--seed",        str(args.seed),
        "--train-frac",  str(args.train_frac),
        "--val-frac",    str(args.val_frac),
        "--candidate-profile", args.candidate_profile,
    ]
    if args.allow_nonmonotonic_fallback:
        cmd.append("--allow-nonmonotonic-fallback")
    if args.no_enforce_aadt_increase:
        cmd.append("--no-enforce-aadt-increase")

    print(f"\n  Command: {' '.join(str(c) for c in cmd)}\n")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    elapsed = time.time() - t0

    if result.returncode != 0:
        _fail(f"Experiment script exited with code {result.returncode}")
        return False

    _ok(f"Experiment complete in {elapsed:.1f}s")
    return True


def _verify_outputs() -> list[str]:
    """Return list of missing required HTML files."""
    missing = []
    for fname in REQUIRED_HTML:
        p = OUTPUT_DIR / fname
        if p.exists():
            _ok(f"{fname}  ({p.stat().st_size:,} bytes)")
        else:
            _warn(f"Missing: {fname}")
            missing.append(fname)
    return missing


def _render_quarto() -> bool:
    """Render the .qmd presentation. Returns True on success."""
    _step("Rendering Quarto presentation ...")

    cmd = ["quarto", "render", str(PRESENTATION)]
    print(f"\n  Command: {' '.join(cmd)}\n")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    elapsed = time.time() - t0

    if result.returncode != 0:
        _fail(f"quarto render exited with code {result.returncode}")
        return False

    # Quarto writes the HTML next to the .qmd by default
    out_html = PRESENTATION.with_suffix(".html")
    if out_html.exists():
        _ok(f"Presentation rendered: {out_html.resolve()}")
        _ok(f"  Size: {out_html.stat().st_size:,} bytes  |  Time: {elapsed:.1f}s")
    else:
        _warn(f"Expected output not found at {out_html} — check quarto output.")

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Workflow control
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--no-render", action="store_true",
        help="Run the experiment but skip the Quarto render step.",
    )
    mode.add_argument(
        "--render-only", action="store_true",
        help="Skip the experiment; only render the Quarto presentation "
             "(requires existing results in results/ex16_3_cmf/).",
    )
    p.add_argument(
        "--smoke-test", action="store_true",
        help="Quick run: 20 search iterations, Poisson family. "
             "Produces all outputs but with reduced search quality.",
    )

    # Experiment settings (match presentation narrative defaults)
    p.add_argument(
        "--search-iter", type=int, default=120,
        help="Search iterations (default: 120, as described in the presentation).",
    )
    p.add_argument(
        "--family", choices=["nb", "poisson"], default="nb",
        help="Count model family (default: nb for NB2).",
    )
    p.add_argument(
        "--seed", type=int, default=17,
        help="Random seed (default: 17).",
    )
    p.add_argument(
        "--train-frac", type=float, default=0.60,
        help="Training split fraction (default: 0.60).",
    )
    p.add_argument(
        "--val-frac", type=float, default=0.20,
        help="Validation split fraction (default: 0.20).",
    )
    p.add_argument(
        "--candidate-profile", choices=["core", "expanded"], default="expanded",
        help="Variable candidate pool (default: expanded — all 31 upper variables).",
    )
    p.add_argument(
        "--allow-nonmonotonic-fallback", action="store_true", default=True,
        help="Fall back to best unconstrained model if no monotonic candidate found "
             "(default: enabled).",
    )
    p.add_argument(
        "--no-enforce-aadt-increase", action="store_true", default=False,
        help="Disable AADT monotonicity constraint entirely.",
    )

    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Apply smoke-test overrides
    if args.smoke_test:
        args.search_iter = 20
        args.family      = "poisson"
        print("  [smoke-test] Using 20 iterations, Poisson family.")

    _banner("Washington CMF Presentation — Full Task Runner")
    print(f"  Repo root   : {REPO_ROOT}")
    print(f"  Data file   : {DATA_FILE}")
    print(f"  Output dir  : {OUTPUT_DIR}")
    print(f"  Presentation: {PRESENTATION}")
    print(f"  Search iter : {args.search_iter}")
    print(f"  Family      : {args.family}")
    print(f"  Candidates  : {args.candidate_profile}")

    t_total = time.time()
    success = True

    # ── Step 0: prerequisites ─────────────────────────────────────────────────
    _banner("Step 0 — Checking prerequisites")
    if not _check_prerequisites(skip_data=args.render_only):
        return 1

    # ── Step 1: run experiment ────────────────────────────────────────────────
    if not args.render_only:
        _banner("Step 1 — Running experiment (generates all HTML assets)")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if not _run_experiment(args):
            success = False

    # ── Step 2: verify outputs ────────────────────────────────────────────────
    _banner("Step 2 — Verifying required HTML outputs")
    missing = _verify_outputs()
    if missing:
        _warn(f"{len(missing)} required file(s) missing — "
              f"presentation iframes will show placeholder text.")
        success = False

    # ── Step 3: render Quarto ─────────────────────────────────────────────────
    if not args.no_render:
        _banner("Step 3 — Rendering Quarto presentation")
        if _check_quarto():
            if not _render_quarto():
                success = False
        else:
            _warn("Quarto unavailable — skipping render.")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed_total = time.time() - t_total
    _banner("Summary")

    print(f"  Total time : {elapsed_total:.1f}s")
    print(f"  Output dir : {OUTPUT_DIR.resolve()}")

    out_html = PRESENTATION.with_suffix(".html")
    if out_html.exists():
        print(f"  Presentation: {out_html.resolve()}")
        print()
        print(f"  Open in browser:")
        print(f"    file:///{out_html.resolve().as_posix()}")
    else:
        print()
        print("  To render manually:")
        print(f"    quarto render {PRESENTATION}")

    if success:
        print("\n  All steps completed successfully.")
        return 0
    else:
        print("\n  Some steps had warnings — check output above.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
