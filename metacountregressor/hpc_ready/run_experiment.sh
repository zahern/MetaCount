#!/bin/bash
# =============================================================================
# MetaCount LC / LC-Tobit Experiment — SLURM submission script
# =============================================================================
#
# OVERVIEW
# --------
# This script submits run_experiment.py to a SLURM HPC cluster.
# Edit the SBATCH directives below (or pass --options on the sbatch command)
# to match your cluster's partition, time limits, and GPU/CPU allocation.
#
# QUICK REFERENCE
# ---------------
# Submit a job:
#   sbatch run_experiment.sh <EXPERIMENT> [OPTIONS...]
#
# EXPERIMENT choices (first positional argument):
#   synthetic_count   -- synthetic LC count data, manual fit 1..2 classes
#   synthetic_tobit   -- synthetic LC Tobit data, manual fit 1..2 classes
#   synthetic_both    -- both of the above in one job
#   count             -- real data, LC NB2, manual mode
#   tobit             -- real data, LC Tobit, manual mode
#   search_count      -- real data, SA structure search, count family
#   search_tobit      -- real data, SA structure search, Tobit family
#
# Full examples:
#   sbatch run_experiment.sh synthetic_both
#   sbatch run_experiment.sh count  --data data/crashes.csv  --n_classes 3
#   sbatch run_experiment.sh tobit  --data data/rates.csv   --n_classes 2 \
#                                   --variables h085,aadt_per_lane,citybound,pct_heavy \
#                                   --membership_vars entry_ramp,exit_ramp
#   sbatch run_experiment.sh search_count --data data/crashes.csv \
#                                          --max_classes 3 --search_iter 5000
#
# INTERACTIVE / LOCAL RUN (no SLURM):
#   bash run_experiment.sh synthetic_both          # runs directly
#   python run_experiment.py --experiment both --synthetic
#
# =============================================================================

# ── SLURM directives ─────────────────────────────────────────────────────────
#SBATCH --job-name=metacount_lc
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=general
# Uncomment if you have a GPU partition:
##SBATCH --partition=gpu
##SBATCH --gres=gpu:1
# Uncomment for email notifications:
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=your.email@institution.edu

# ── Environment setup ────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate your conda / venv environment.
# Edit this block to match your HPC environment:
# ------------------------------------------------
if [ -n "${CONDA_PREFIX:-}" ]; then
    echo "[env] Using active conda env: $CONDA_PREFIX"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate metacount 2>/dev/null || conda activate base
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate metacount 2>/dev/null || conda activate base
elif [ -f "$SCRIPT_DIR/../.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/../.venv/bin/activate"
    echo "[env] Using local .venv"
else
    echo "[env] No conda/venv detected — using system Python"
fi

echo "[env] Python: $(which python)"
echo "[env] JAX: $(python -c 'import jax; print(jax.__version__)' 2>/dev/null || echo 'not found')"
echo "[env] Host: $(hostname)  Cores: ${SLURM_CPUS_PER_TASK:-$(nproc)}"
echo ""

# ── Parse positional experiment argument ────────────────────────────────────
EXPERIMENT="${1:-synthetic_both}"
shift || true          # remove first arg so remaining are passed through

# ── JAX thread tuning (adjust to match --cpus-per-task) ─────────────────────
N_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${N_THREADS}"
export OMP_NUM_THREADS="${N_THREADS}"
export MKL_NUM_THREADS="${N_THREADS}"

echo "=== MetaCount experiment: ${EXPERIMENT} ==="
echo "=== Start: $(date) ==="
echo ""

# ── Dispatch ────────────────────────────────────────────────────────────────
case "${EXPERIMENT}" in

    # ------------------------------------------------------------------
    # Synthetic quick-checks (no data file needed)
    # ------------------------------------------------------------------
    synthetic_count)
        python run_experiment.py \
            --experiment lc_count \
            --synthetic \
            --n_classes 2 \
            --membership_vars z1,z2 \
            --output_dir results/synthetic \
            --tag count \
            "$@"
        ;;

    synthetic_tobit)
        python run_experiment.py \
            --experiment lc_tobit \
            --synthetic \
            --n_classes 2 \
            --membership_vars z1,z2 \
            --output_dir results/synthetic \
            --tag tobit \
            "$@"
        ;;

    synthetic_both)
        python run_experiment.py \
            --experiment both \
            --synthetic \
            --n_classes 2 \
            --membership_vars z1,z2 \
            --output_dir results/synthetic \
            --tag both \
            "$@"
        ;;

    # ------------------------------------------------------------------
    # Real data — manual model fits
    # Pass --data, --id_col, --y_col, --variables, etc. via "$@"
    # ------------------------------------------------------------------
    count)
        python run_experiment.py \
            --experiment lc_count \
            --mode manual \
            --n_classes 2 \
            --output_dir results/count \
            "$@"
        ;;

    tobit)
        python run_experiment.py \
            --experiment lc_tobit \
            --mode manual \
            --n_classes 2 \
            --output_dir results/tobit \
            "$@"
        ;;

    both)
        python run_experiment.py \
            --experiment both \
            --mode manual \
            --n_classes 2 \
            --output_dir results/both \
            "$@"
        ;;

    # ------------------------------------------------------------------
    # Automated SA specification search
    # ------------------------------------------------------------------
    search_count)
        python run_experiment.py \
            --experiment lc_count \
            --mode search \
            --max_classes 3 \
            --search_iter 3000 \
            --output_dir results/search_count \
            "$@"
        ;;

    search_tobit)
        python run_experiment.py \
            --experiment lc_tobit \
            --mode search \
            --max_classes 2 \
            --search_iter 3000 \
            --output_dir results/search_tobit \
            "$@"
        ;;

    # ------------------------------------------------------------------
    # Help / unknown
    # ------------------------------------------------------------------
    help|--help|-h)
        python run_experiment.py --help
        exit 0
        ;;

    *)
        echo "ERROR: Unknown experiment '${EXPERIMENT}'"
        echo ""
        echo "Valid choices:"
        echo "  synthetic_count  synthetic_tobit  synthetic_both"
        echo "  count  tobit  both"
        echo "  search_count  search_tobit"
        echo ""
        echo "Run:  bash run_experiment.sh help   for full usage."
        exit 1
        ;;
esac

echo ""
echo "=== Done: $(date) ==="
