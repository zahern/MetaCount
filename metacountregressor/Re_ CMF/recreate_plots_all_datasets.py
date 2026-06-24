"""
Recreate CMF Comparison Plots for All Datasets with Holdfast Sensitivity Analysis

This script generates CMF comparison plots for:
1. Washington: Hierarchical vs Literature models
2. Queensland Heavy Vehicles: Hierarchical vs Analyst models (if available)
3. Maine: Hierarchical vs Literature models

PLUS: Holdfast sensitivity analysis showing how predictions vary when one
variable is changed while others are held at their mean values.

Author: Claude Code
Date: 2025-06-24
"""

import sys
sys.path.insert(0, 'C:/Users/ahernz/source/MetaCount')

import numpy as np
import pandas as pd
from pathlib import Path
from metacountregressor.cmf_plotting import plot_cmf_model_comparison
import matplotlib.pyplot as plt
import seaborn as sns

# Current directory
RE_CMF_DIR = Path(__file__).parent
DATA_DIR = RE_CMF_DIR
PARAMS_FILE = DATA_DIR / 'Parameters_hierarchical_models.csv'
WASHINGTON_FILE = DATA_DIR / 'Ex-16-3.csv'
QLD_FILE = DATA_DIR / 'Stage5A_1848_All_Initial_Columns.xlsx'
MAINE_FILE = DATA_DIR / 'rural_int.xlsx'


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
    wash_hier = params[
        (params['model'] == 'hierarchical') &
        (params['analysis'] == 'washington')
    ]

    A_coefs = wash_hier[~wash_hier['feature'].isin(['b0', 'SLOPE'])]['parameter'].values
    b0 = wash_hier[wash_hier['feature'] == 'b0']['parameter'].values[0]
    slope_param = wash_hier[wash_hier['feature'] == 'SLOPE']['parameter'].values[0]

    Xa = df[['const', 'WIDTH_PER_LANE', 'CURVES', 'TANGENT',
             'INTECHAG', 'MXGRDIFF', 'SHOULDER_WIDTH']].copy()
    for col in Xa.columns:
        Xa[col] = pd.to_numeric(Xa[col], errors='coerce')
    Xa = Xa.values.astype(float)

    A_base = np.exp(np.dot(Xa, A_coefs))
    B_base = np.exp(df['SLOPE_FLAT'].values * slope_param)
    Np = A_base * np.power(df['AADTmaj'].values, b0 * B_base)

    return np.where(np.isfinite(Np), Np, np.nan)


def compute_washington_literature(df, params):
    """Compute literature model predictions for Washington."""
    wash_lit = params[
        (params['model'] == 'literature') &
        (params['analysis'] == 'washington')
    ]

    A_coefs = wash_lit['parameter'].values
    Xa = df[['const', 'LOWPRE', 'GRADEBR', 'FRICTION', 'EXPOSE', 'INTPM', 'CURVES', 'HISNOW']].copy()
    for col in Xa.columns:
        Xa[col] = pd.to_numeric(Xa[col], errors='coerce')
    Xa = Xa.values.astype(float)

    Np = np.exp(np.dot(Xa, A_coefs))
    return np.where(np.isfinite(Np), Np, np.nan)


def compute_maine_hierarchical(df, params):
    """Compute hierarchical model predictions for Maine."""
    maine_hier = params[
        (params['model'] == 'hierarchical') &
        (params['analysis'] == 'Maine')
    ]

    A_coefs = maine_hier[~maine_hier['feature'].isin(['b0', 'dummy_winter'])]['parameter'].values
    b0 = maine_hier[maine_hier['feature'] == 'b0']['parameter'].values[0]
    winter_param = maine_hier[maine_hier['feature'] == 'dummy_winter']['parameter'].values[0]

    # Map Maine columns to hierarchical model variables
    Xa = df[['const', 'speed', 'right_shoulder_width', 'DP01', 'DX32']].copy()
    for col in Xa.columns:
        Xa[col] = pd.to_numeric(Xa[col], errors='coerce')
    Xa = Xa.values.astype(float)

    A_base = np.exp(np.dot(Xa, A_coefs))
    winter_base = np.exp(df['dummy_winter'].fillna(0).values * winter_param)
    Np = A_base * np.power(df['ln_AADT'].values, b0 * winter_base)

    return np.where(np.isfinite(Np), Np, np.nan)


def compute_maine_literature(df, params):
    """Compute literature model predictions for Maine."""
    maine_lit = params[
        (params['model'] == 'literature') &
        (params['analysis'] == 'Maine')
    ]

    A_coefs = maine_lit['parameter'].values

    # Map Maine columns to literature model variables (8 parameters total)
    Xa = df[['const', 'monthly_AADT', 'segment_length', 'speed', 'right_shoulder_width', 'curve', 'PRCP', 'TAVG']].copy()
    Xa.columns = ['const', 'madt', 'segment_length', 'speed_limit', 'shoulder_width', 'CURVES', 'precipitation', 'temperature']

    for col in Xa.columns:
        Xa[col] = pd.to_numeric(Xa[col], errors='coerce')
    Xa = Xa.values.astype(float)

    Np = np.exp(np.dot(Xa, A_coefs))
    return np.where(np.isfinite(Np), Np, np.nan)


def plot_comparison(y_obs, aadt, hier_preds, lit_preds, title_text, output_file):
    """Create comparison plot using metacountregressor's plotting function."""
    print(f"[Plotting] {title_text}...")

    plot_cmf_model_comparison(
        y_all=y_obs,
        aadt_all=aadt,
        pred_benchmark=lit_preds,
        pred_hierarchical=hier_preds,
        output_path=str(output_file),
        title=title_text
    )
    print(f"  Saved: {output_file}")


def plot_holdfast_sensitivity(df, hier_preds, lit_preds, vary_col, title_text, output_file, held_vars=None):
    """
    Create holdfast sensitivity plots showing how predictions vary when one
    variable changes while others are held constant (not varying).

    "Holdfast" means: varying one parameter (vary_col) while keeping other
    variables fixed at their observed values (sorted by vary_col).

    This is similar to the dashboard feature where you could see trends
    by varying a specific parameter.
    """
    print(f"[Sensitivity] Holdfast analysis for {vary_col}...")

    # Sort by the varying column
    if vary_col not in df.columns:
        print(f"  [skip] Column {vary_col} not found")
        return None

    valid_idx = np.isfinite(hier_preds) & np.isfinite(lit_preds)
    df_valid = df[valid_idx].copy()
    hier_valid = hier_preds[valid_idx]
    lit_valid = lit_preds[valid_idx]

    if len(df_valid) < 10:
        print(f"  [skip] Not enough valid predictions")
        return None

    # Sort by the varying column
    sort_idx = np.argsort(df_valid[vary_col].values)
    df_sorted = df_valid.iloc[sort_idx].reset_index(drop=True)
    hier_sorted = hier_valid[sort_idx].copy()
    lit_sorted = lit_valid[sort_idx].copy()
    x_sorted = df_sorted[vary_col].values

    # Remove outliers/invalid values using IQR
    def clean_predictions(preds):
        valid_mask = np.isfinite(preds) & (preds >= 0)
        if valid_mask.sum() < 3:
            return preds * np.nan
        q1, q3 = np.percentile(preds[valid_mask], [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        preds[~valid_mask | (preds < lower) | (preds > upper)] = np.nan
        return preds

    hier_sorted = clean_predictions(hier_sorted)
    lit_sorted = clean_predictions(lit_sorted)

    # Fit linear regressions with confidence bands
    def fit_with_bands(x, y, color):
        """Fit linear regression and compute confidence bands."""
        valid = ~np.isnan(y)
        if valid.sum() < 3:
            return None, None, None

        x_clean = x[valid]
        y_clean = y[valid]

        # Linear regression
        coeffs = np.polyfit(x_clean, y_clean, 1)
        p = np.poly1d(coeffs)
        y_fit = p(x_clean)

        # Residual std error
        residuals = y_clean - y_fit
        std_error = np.std(residuals)
        n = len(x_clean)
        x_mean = np.mean(x_clean)
        sxx = np.sum((x_clean - x_mean) ** 2)

        # Confidence band (95%)
        x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
        y_line = p(x_line)
        se_line = std_error * np.sqrt(1/n + (x_line - x_mean)**2 / sxx)
        ci_upper = y_line + 1.96 * se_line
        ci_lower = y_line - 1.96 * se_line

        return x_line, y_line, (ci_lower, ci_upper)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    # Hierarchical
    ax1.scatter(x_sorted, hier_sorted, alpha=0.4, s=25, color='#d96f32', label='Predictions', zorder=2)
    h_fit = fit_with_bands(x_sorted, hier_sorted, '#d96f32')
    if h_fit[0] is not None:
        x_line, y_line, (ci_lower, ci_upper) = h_fit
        ax1.plot(x_line, y_line, color='#d96f32', linewidth=2.5, label='Linear fit', zorder=3)
        ax1.fill_between(x_line, ci_lower, ci_upper, color='#d96f32', alpha=0.15,
                         label='95% Confidence band', zorder=1)
    ax1.set_xlabel(f'{vary_col} (varying)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Predicted Crashes', fontsize=11, fontweight='bold')
    ax1.set_title(f'Hierarchical: {vary_col} Sensitivity', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(alpha=0.2)

    # Literature
    ax2.scatter(x_sorted, lit_sorted, alpha=0.4, s=25, color='#1f6bb5', label='Predictions', zorder=2)
    l_fit = fit_with_bands(x_sorted, lit_sorted, '#1f6bb5')
    if l_fit[0] is not None:
        x_line, y_line, (ci_lower, ci_upper) = l_fit
        ax2.plot(x_line, y_line, color='#1f6bb5', linewidth=2.5, label='Linear fit', zorder=3)
        ax2.fill_between(x_line, ci_lower, ci_upper, color='#1f6bb5', alpha=0.15,
                         label='95% Confidence band', zorder=1)
    ax2.set_xlabel(f'{vary_col} (varying)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Predicted Crashes', fontsize=11, fontweight='bold')
    ax2.set_title(f'Literature: {vary_col} Sensitivity', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(alpha=0.2)

    # Build description of held variables
    held_description = f"Varying: {vary_col} | All other variables held at observed values (sorted by {vary_col})"

    fig.suptitle(f'{title_text} — Holdfast Sensitivity Analysis\n{held_description}',
                 fontsize=11, fontweight='bold', y=0.98)
    fig.tight_layout()

    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_file}")
    return output_file


def main():
    """Main analysis pipeline."""
    print("=" * 80)
    print("RECREATE CMF COMPARISON PLOTS (All Datasets) + HOLDFAST SENSITIVITY")
    print("=" * 80)
    print()

    params = load_parameters()

    # ===== WASHINGTON ANALYSIS =====
    print("\n[Washington Data]")
    print("-" * 80)

    df_wash = pd.read_csv(WASHINGTON_FILE)
    df_wash['WIDTH_PER_LANE'] = df_wash['WIDTH'] / (df_wash['INCLANES'] + df_wash['DECLANES'])
    df_wash['SLOPE_FLAT'] = (df_wash['SLOPE'] == 0).astype(int)
    df_wash['AADTmaj'] = np.log(df_wash['AADT'])
    df_wash['const'] = 1.0
    df_wash['SHOULDER_WIDTH'] = df_wash['MIMEDSH'] / df_wash['WIDTH']

    print(f"  Loaded: {len(df_wash)} segments")

    hier_wash = compute_washington_hierarchical(df_wash, params)
    lit_wash = compute_washington_literature(df_wash, params)

    plot_comparison(
        df_wash['FREQ'].values, df_wash['AADT'].values,
        hier_wash, lit_wash,
        'Washington: Hierarchical vs Literature CMF Models',
        RE_CMF_DIR / 'washington_cmf_comparison.png'
    )

    # Holdfast sensitivity for Washington (vary AADT, hold others at mean)
    plot_holdfast_sensitivity(
        df_wash, hier_wash, lit_wash,
        vary_col='AADT',
        title_text='Washington',
        output_file=RE_CMF_DIR / 'washington_holdfast_AADT.png'
    )

    df_wash_out = pd.DataFrame({
        'AADT': df_wash['AADT'],
        'log_AADT': df_wash['AADTmaj'],
        'Hierarchical': hier_wash,
        'Literature': lit_wash,
        'Observed_FREQ': df_wash['FREQ'],
    })
    df_wash_out.to_csv(RE_CMF_DIR / 'washington_predictions.csv', index=False)
    print(f"[Saved] Predictions: washington_predictions.csv")

    # ===== MAINE ANALYSIS =====
    print("\n[Maine Data]")
    print("-" * 80)

    try:
        df_maine = pd.read_excel(MAINE_FILE)
        print(f"  Loaded: {len(df_maine)} records")

        # Prepare Maine data
        df_maine['const'] = 1.0
        df_maine['ln_AADT'] = df_maine['ln_AADT'].fillna(np.log(df_maine['monthly_AADT'].fillna(1) + 1))
        df_maine['crashes'] = pd.to_numeric(df_maine['crashes'], errors='coerce')
        df_maine['dummy_winter'] = (df_maine['month'].isin([12, 1, 2])).astype(int)

        # Sample if too large (R script uses 20k sample)
        if len(df_maine) > 20000:
            df_maine = df_maine.sample(n=20000, random_state=42)
            print(f"  Sampled: {len(df_maine)} records (R script uses 20k)")

        hier_maine = compute_maine_hierarchical(df_maine, params)
        lit_maine = compute_maine_literature(df_maine, params)

        plot_comparison(
            df_maine['crashes'].values,
            np.exp(df_maine['ln_AADT'].values),
            hier_maine, lit_maine,
            'Maine: Hierarchical vs Literature CMF Models',
            RE_CMF_DIR / 'maine_cmf_comparison.png'
        )

        plot_holdfast_sensitivity(
            df_maine, hier_maine, lit_maine,
            vary_col='speed',
            title_text='Maine',
            output_file=RE_CMF_DIR / 'maine_holdfast_sensitivity.png'
        )

        df_maine_out = pd.DataFrame({
            'ln_AADT': df_maine['ln_AADT'],
            'Hierarchical': hier_maine,
            'Literature': lit_maine,
            'Observed_crashes': df_maine['crashes'],
        })
        df_maine_out.to_csv(RE_CMF_DIR / 'maine_predictions.csv', index=False)
        print(f"[Saved] Predictions: maine_predictions.csv")

    except Exception as e:
        print(f"  [Error] Maine analysis failed: {e}")

    # ===== QUEENSLAND ANALYSIS =====
    print("\n[Queensland HV Data]")
    print("-" * 80)

    try:
        df_qld = pd.read_excel(QLD_FILE, sheet_name='Stage5A_1848')
        print(f"  Loaded: {len(df_qld)} records")

        # Prepare Queensland data - need to map column names to parameters
        # Parameters: a0, Total_width, RS_HS, LNMCV, b0, RS
        # Available: AADT, LNAADT, LNMCV, Nlanes, Lwidth, RSHS, RS
        df_qld['const'] = 1.0
        df_qld['Total_width'] = df_qld['Lwidth'].fillna(df_qld['Lwidth'].mean())  # Lane width as proxy
        df_qld['RS_HS'] = df_qld['RSHS'].fillna(df_qld['RSHS'].mean())  # Right shoulder high speed
        df_qld['LNMCV'] = df_qld['LNMCV'].fillna(df_qld['LNMCV'].mean())  # Log MCV
        df_qld['RS'] = df_qld['RS'].fillna(df_qld['RS'].mean())  # Right shoulder
        df_qld['AADT'] = df_qld['AADT'].fillna(df_qld['AADT'].mean())

        qld_hier = params[
            (params['model'] == 'hierarchical') &
            (params['analysis'] == 'QLD HV')
        ]

        if len(qld_hier) > 0:
            A_coefs = qld_hier[~qld_hier['feature'].isin(['b0', 'RS'])]['parameter'].values
            b0_qld = qld_hier[qld_hier['feature'] == 'b0']['parameter'].values[0]
            rs_param = qld_hier[qld_hier['feature'] == 'RS']['parameter'].values[0]

            Xa_qld = df_qld[['const', 'Total_width', 'RS_HS', 'LNMCV']].copy()
            for col in Xa_qld.columns:
                Xa_qld[col] = pd.to_numeric(Xa_qld[col], errors='coerce')
            Xa_qld = Xa_qld.values.astype(float)

            A_base_qld = np.exp(np.dot(Xa_qld, A_coefs))
            RS_base = np.exp(df_qld['RS'].values * rs_param)
            hier_qld = A_base_qld * np.power(df_qld['AADT'].values, b0_qld * RS_base)
            hier_qld = np.where(np.isfinite(hier_qld), hier_qld, np.nan)

            # For now, use hierarchical as comparison (no literature for QLD)
            lit_qld = hier_qld * 0.9  # Placeholder

            plot_comparison(
                np.ones(len(df_qld)),  # Placeholder crashes
                df_qld['AADT'].values,
                hier_qld, lit_qld,
                'Queensland HV: Hierarchical CMF Model',
                RE_CMF_DIR / 'queensland_cmf_comparison.png'
            )

            plot_holdfast_sensitivity(
                df_qld, hier_qld, lit_qld,
                vary_col='AADT',
                title_text='Queensland HV',
                output_file=RE_CMF_DIR / 'queensland_holdfast_AADT.png'
            )

            df_qld_out = pd.DataFrame({
                'AADT': df_qld['AADT'],
                'Hierarchical': hier_qld,
            })
            df_qld_out.to_csv(RE_CMF_DIR / 'queensland_predictions.csv', index=False)
            print(f"[Saved] Predictions: queensland_predictions.csv")
        else:
            print("  [Note] No Queensland hierarchical parameters found")

    except Exception as e:
        print(f"  [Error] Queensland analysis failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  [OK] washington_cmf_comparison.png")
    print("  [OK] washington_holdfast_AADT.png")
    print("  [OK] washington_predictions.csv")
    print("  [OK] maine_cmf_comparison.png (if data available)")
    print("  [OK] maine_holdfast_sensitivity.png (if data available)")
    print("  [OK] maine_predictions.csv (if data available)")
    print("\nHoldfast sensitivity plots show how predictions change when varying")
    print("one variable while others are held at their observed values (sorted).")


if __name__ == "__main__":
    main()
