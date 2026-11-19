from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metacountregressor import (
    ExperimentBuilder,
    load_example16_3_model_data,
)


def _to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        # Fallback when tabulate is unavailable.
        cols = list(df.columns)
        header = "| " + " | ".join(str(c) for c in cols) + " |"
        divider = "| " + " | ".join(["---"] * len(cols)) + " |"
        rows = [
            "| " + " | ".join(str(row[c]) for c in cols) + " |"
            for _, row in df.iterrows()
        ]
        return "\n".join([header, divider] + rows)


def build_and_fit(fit_draws: int) -> tuple[dict, pd.DataFrame]:
    df = load_example16_3_model_data().copy()

    # Stabilize coefficient scale for reporting: z-score continuous covariates.
    for col in ["AADT", "SPEED", "CURVES", "AVEPRE"]:
        mu = float(df[col].mean())
        sd = float(df[col].std(ddof=0))
        denom = sd if sd > 0 else 1.0
        df[f"{col}_Z"] = (df[col] - mu) / denom

    builder = ExperimentBuilder(
        df=df,
        id_col="ID",
        y_col="FREQ",
        offset_col="OFFSET",
        group_id_col="FC",
    )

    manual_spec = builder.make_manual_spec(
        # OFFSET already carries exposure (log(AADT*LENGTH*...)).
        # Keep covariates orthogonal-ish and scaled for interpretable magnitudes.
        fixed_terms=["AADT_Z", "SPEED_Z", "URB"],
        rdm_terms=["CURVES_Z:normal"],
        hetro_in_means=["AVEPRE_Z"],
        dispersion=1,
        latent_classes=1,
    )

    fit = builder.fit_manual_model(
        manual_spec=manual_spec,
        model="nb",
        R=fit_draws,
    )

    coef_df = builder.print_coefficients(fit).copy()
    coef_df["Estimate"] = coef_df["Estimate"].round(4)

    return fit, coef_df


def export_assets(fit: dict, coef_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fit_overview = pd.DataFrame(
        [
            {"Metric": "Pipeline", "Value": "Task automated manual NB fit (scaled covariates)"},
            {"Metric": "Refit model", "Value": fit["spec"].model.upper()},
            {"Metric": "Offset handling", "Value": "OFFSET added directly to eta (coefficient fixed at 1)"},
            {"Metric": "Rows used in refit", "Value": int(len(fit["data"]["y"]))},
            {"Metric": "Parameter count", "Value": int(len(fit["result"].params))},
        ]
    )

    overview_md = _to_markdown(fit_overview)
    coef_md = _to_markdown(coef_df)

    (output_dir / "quickstart_fit_overview.md").write_text(overview_md, encoding="utf-8")
    (output_dir / "quickstart_fit_coefficients.md").write_text(coef_md, encoding="utf-8")
    (output_dir / "quickstart_fit_coefficients.html").write_text(
        coef_df.to_html(index=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a quickstart manual fit and export coefficient assets for slides."
    )
    parser.add_argument("--fit-draws", type=int, default=500, help="Halton draws used for final refit.")
    parser.add_argument(
        "--output-dir",
        default="results/slide_assets",
        help="Directory where markdown/html assets will be written.",
    )

    args = parser.parse_args()

    print("Running quickstart manual model refit...")
    fit, coef_df = build_and_fit(fit_draws=args.fit_draws)

    output_dir = Path(args.output_dir)
    export_assets(fit=fit, coef_df=coef_df, output_dir=output_dir)

    print("Pipeline: manual NB fit with random curvature effect")
    print(f"Assets written to: {output_dir.resolve()}")
    print("Created files:")
    print("- quickstart_fit_overview.md")
    print("- quickstart_fit_coefficients.md")
    print("- quickstart_fit_coefficients.html")


if __name__ == "__main__":
    main()
