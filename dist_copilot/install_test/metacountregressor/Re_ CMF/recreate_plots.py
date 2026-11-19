"""
Recreate CMF Comparison Plots from R Script

This script recreates the three comparison plots from Performance_approaches.R
using metacountregressor's built-in CMF plotting functions:

1. Washington: Hierarchical vs Literature models
2. Queensland Heavy Vehicles: Hierarchical vs Analyst models
3. Maine: Hierarchical vs Literature models

Uses the package's plot_cmf_model_comparison() function for consistent styling
and professional output.

Author: Claude Code
Date: 2025-06-24
"""

import sys
sys.path.insert(0, 'C:/Users/ahernz/source/MetaCount')

import numpy as np
import pandas as pd
from pathlib import Path
from metacountregressor.cmf_plotting import plot_cmf_model_comparison

# Current directory
RE_CMF_DIR = Path(__file__).parent
DATA_DIR = RE_CMF_DIR
PARAMS_FILE = DATA_DIR / 'Parameters_hierarchical_models.csv'
WASHINGTON_FILE = DATA_DIR / 'Ex-16-3.csv'

# For QLD and Maine - these need to be sourced from elsewhere
# For now, we'll focus on Washington which we have data for


def load_parameters():
    """Load parameters from CSV."""
    print("[Loading] Parameters...")
    params = pd.read_csv(PARAMS_FILE, sep=';')
    params.columns = params.columns.str.strip()
    params['parameter'] = params['parameter'].astype(str).str.replace(',', '.')
    params['parameter'] = pd.to_numeric(params['parameter'])
    return params


def compute_washington_hierarchical(df, params):
    """Compute hierarchical model predictions for Washington."""
    # Get Washington hierarchical parameters
    wash_hier = params[
        (params['model'] == 'hierarchical') &
        (params['analysis'] == 'washington')
    ]

    # Extract A coefficients (first 7)
    A_coefs = wash_hier[~wash_hier['feature'].isin(['b0', 'SLOPE'])]['parameter'].values
    b0 = wash_hier[wash_hier['feature'] == 'b0']['parameter'].values[0]
    slope_param = wash_hier[wash_hier['feature'] == 'SLOPE']['parameter'].values[0]

    # Design matrix
    Xa = df[['const', 'WIDTH_PER_LANE', 'CURVES', 'TANGENT',
             'INTECHAG', 'MXGRDIFF', 'SHOULDER_WIDTH']].copy()
    for col in Xa.columns:
        Xa[col] = pd.to_numeric(Xa[col], errors='coerce')
    Xa = Xa.values.astype(float)

    # Compute predictions
    A_base = np.exp(np.dot(Xa, A_coefs))
    B_base = np.exp(df['SLOPE_FLAT'].values * slope_param)
    # Paper form (Eq 37-39): Np = A(z1) * x^{B(z2)} with x = AADT (exposure).
    Np = A_base * np.power(df['AADT'].values.astype(float), b0 * B_base)

    return np.where(np.isfinite(Np), Np, np.nan)


def compute_washington_literature(df, params):
    """Compute literature model predictions for Washington."""
    # Get Washington literature parameters
    wash_lit = params[
        (params['model'] == 'literature') &
        (params['analysis'] == 'washington')
    ]

    # Extract parameters
    A_coefs = wash_lit['parameter'].values

    # Design matrix - match R script
    Xa = df[['const', 'LOWPRE', 'GRADEBR', 'FRICTION', 'EXPOSE', 'INTPM', 'CURVES', 'HISNOW']].copy()
    for col in Xa.columns:
        Xa[col] = pd.to_numeric(Xa[col], errors='coerce')
    Xa = Xa.values.astype(float)

    # Compute predictions: Np = exp(Xa @ A)
    Np = np.exp(np.dot(Xa, A_coefs))
    return np.where(np.isfinite(Np), Np, np.nan)


def plot_comparison(y_obs, aadt, hier_preds, lit_preds,
                    title_text, output_file):
    """
    Create comparison plot using metacountregressor's plotting function.

    Uses the package's plot_cmf_model_comparison which provides:
    - Professional 2x2 layout with scatter + smooth curves
    - Consistent color scheme (blue for benchmark, orange for hierarchical)
    - Residual analysis subplots
    - Publication-ready output
    """
    print(f"[Plotting] {title_text}...")

    # Use the package's built-in plotting function
    # Note: This function expects (benchmark, hierarchical) but we'll use
    # (literature, hierarchical) for consistency with the R script
    plot_cmf_model_comparison(
        y_all=y_obs,
        aadt_all=aadt,
        pred_benchmark=lit_preds,      # Use literature as "benchmark"
        pred_hierarchical=hier_preds,
        output_path=str(output_file),
        title=title_text
    )
    print(f"  Saved: {output_file}")


def main():
    """Main analysis pipeline."""
    print("=" * 80)
    print("RECREATE CMF COMPARISON PLOTS (Python from R Script)")
    print("=" * 80)
    print()

    # Load parameters
    params = load_parameters()

    # ===== WASHINGTON ANALYSIS =====
    print("\n[Washington Data]")
    print("-" * 80)

    # Load and prepare data
    print("[Loading] Ex-16-3.csv...")
    df_wash = pd.read_csv(WASHINGTON_FILE)

    # Prepare variables (matching R script exactly)
    df_wash['WIDTH_PER_LANE'] = df_wash['WIDTH'] / (df_wash['INCLANES'] + df_wash['DECLANES'])
    df_wash['SLOPE_FLAT'] = (df_wash['SLOPE'] == 0).astype(int)
    df_wash['AADTmaj'] = np.log(df_wash['AADT'])
    df_wash['const'] = 1.0
    df_wash['SHOULDER_WIDTH'] = df_wash['MIMEDSH'] / df_wash['WIDTH']
    df_wash['TOTAL_LANES'] = df_wash['INCLANES'] + df_wash['DECLANES']

    print(f"  Loaded: {len(df_wash)} segments")

    # Compute predictions
    print("[Computing] Hierarchical predictions...")
    hier_wash = compute_washington_hierarchical(df_wash, params)

    print("[Computing] Literature predictions...")
    lit_wash = compute_washington_literature(df_wash, params)

    # Plot using package's CMF plotting function
    plot_comparison(
        y_obs=df_wash['FREQ'].values,
        aadt=df_wash['AADT'].values,
        hier_preds=hier_wash,
        lit_preds=lit_wash,
        title_text='Washington: Hierarchical vs Literature CMF Models',
        output_file=RE_CMF_DIR / 'washington_cmf_comparison.png'
    )

    # Save predictions to CSV
    df_wash_out = pd.DataFrame({
        'AADT': df_wash['AADT'],
        'log_AADT': df_wash['AADTmaj'],
        'Hierarchical': hier_wash,
        'Literature': lit_wash,
        'Observed_FREQ': df_wash['FREQ'],
    })
    csv_path = RE_CMF_DIR / 'washington_predictions.csv'
    df_wash_out.to_csv(csv_path, index=False)
    print(f"[Saved] Predictions: {csv_path}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print(f"  - washington_cmf_comparison.png")
    print(f"  - washington_predictions.csv")
    print("\nNote: Queensland and Maine datasets are not included in this directory.")
    print("To add them, update paths in this script.")


if __name__ == "__main__":
    main()
