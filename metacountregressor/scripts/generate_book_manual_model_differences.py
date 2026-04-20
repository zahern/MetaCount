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
    compare_models,
    load_book_latent_class_spec,
    load_book_nb_baseline_spec,
    load_example16_3_model_data,
)


SPEC_KEYS = [
    "fixed_terms",
    "rdm_terms",
    "rdm_cor_terms",
    "grouped_terms",
    "hetro_in_means",
    "zi_terms",
    "membership_terms",
    "dispersion",
    "latent_classes",
]


def _to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        cols = list(df.columns)
        header = "| " + " | ".join(str(c) for c in cols) + " |"
        divider = "| " + " | ".join(["---"] * len(cols)) + " |"
        rows = [
            "| " + " | ".join(str(row[c]) for c in cols) + " |"
            for _, row in df.iterrows()
        ]
        return "\n".join([header, divider] + rows)


def _fmt_list(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        if not value:
            return "(none)"
        return ", ".join(str(v) for v in value)
    return str(value)


def _list_difference(a: list, b: list) -> str:
    a_set = set(a)
    b_set = set(b)
    only_a = sorted(a_set - b_set)
    only_b = sorted(b_set - a_set)

    chunks = []
    if only_a:
        chunks.append("NB only: " + ", ".join(only_a))
    if only_b:
        chunks.append("LC only: " + ", ".join(only_b))
    if not chunks:
        return "same"
    return " | ".join(chunks)


def build_spec_difference_table(nb_spec: dict, lc_spec: dict) -> pd.DataFrame:
    rows = []
    for key in SPEC_KEYS:
        nb_val = nb_spec.get(key)
        lc_val = lc_spec.get(key)

        if isinstance(nb_val, list) and isinstance(lc_val, list):
            diff = _list_difference(nb_val, lc_val)
        else:
            diff = "same" if nb_val == lc_val else f"NB={nb_val} -> LC={lc_val}"

        rows.append(
            {
                "Section": key,
                "NB baseline (book)": _fmt_list(nb_val),
                "NB latent-class (book)": _fmt_list(lc_val),
                "Difference": diff,
            }
        )

    return pd.DataFrame(rows)


def build_coefficient_difference_table(
    coef_nb: pd.DataFrame,
    coef_lc: pd.DataFrame,
) -> pd.DataFrame:
    merged = coef_nb.merge(
        coef_lc,
        on=["Parameter", "Type"],
        how="outer",
    )

    merged["Estimate_baseline"] = pd.to_numeric(merged["Estimate_baseline"], errors="coerce")
    merged["Estimate_latent_class"] = pd.to_numeric(merged["Estimate_latent_class"], errors="coerce")
    merged["Delta_LC_minus_baseline"] = (
        merged["Estimate_latent_class"] - merged["Estimate_baseline"]
    )

    merged = merged.sort_values(["Type", "Parameter"], na_position="last").reset_index(drop=True)
    return merged


def build_slide_summary(metrics_df: pd.DataFrame, spec_diff_df: pd.DataFrame) -> str:
    changed = spec_diff_df[spec_diff_df["Difference"] != "same"].copy()
    if changed.empty:
        changed = spec_diff_df.head(5)

    metrics_md = _to_markdown(metrics_df)
    changed_md = _to_markdown(changed[["Section", "Difference"]])

    return (
        "Book manual model comparison (from fitted specifications)\n\n"
        "Fit metrics\n\n"
        f"{metrics_md}\n\n"
        "Key structural differences\n\n"
        f"{changed_md}"
    )


def _fit_with_lc_fallback(
    builder: ExperimentBuilder,
    spec: dict,
    fit_draws: int,
) -> tuple[dict, str]:
    """Fit a latent-class manual spec and fall back to fixed-only LC when needed.

    Some JAX latent-class paths can fail on specific random-effect structures.
    This keeps the task robust while still producing a comparable LC model.
    """
    try:
        fit = builder.fit_manual_model(manual_spec=spec, model="nb", R=fit_draws)
        return fit, "book latent-class spec"
    except Exception as exc:
        fallback = dict(spec)
        fallback["rdm_terms"] = []
        fallback["rdm_cor_terms"] = []
        fallback["grouped_terms"] = []
        fit = builder.fit_manual_model(manual_spec=fallback, model="nb", R=fit_draws)
        reason = (
            "fallback latent-class spec (fixed-only): "
            f"original random-effect LC fit failed with {type(exc).__name__}"
        )
        return fit, reason


def _safe_coef_table(builder: ExperimentBuilder, fit_result: dict, estimate_col: str) -> tuple[pd.DataFrame, str]:
    """Return coefficient table when supported by the fit layout."""
    try:
        coef = builder.print_coefficients(fit_result).rename(columns={"Estimate": estimate_col})
        return coef, "ok"
    except Exception as exc:
        empty = pd.DataFrame(columns=["Parameter", "Type", estimate_col])
        return empty, f"unavailable ({type(exc).__name__})"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit book manual specs (NB baseline vs NB latent-class) and export differences."
    )
    parser.add_argument("--fit-draws", type=int, default=120, help="Halton draws for each model fit.")
    parser.add_argument(
        "--output-dir",
        default="results/slide_assets",
        help="Output directory for markdown/html comparison assets.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_example16_3_model_data()
    builder = ExperimentBuilder(
        df=df,
        id_col="ID",
        y_col="FREQ",
        offset_col="OFFSET",
        group_id_col="FC",
    )

    nb_spec = load_book_nb_baseline_spec()
    lc_spec = load_book_latent_class_spec()

    print("Fitting book NB baseline model...")
    fit_nb = builder.fit_manual_model(manual_spec=nb_spec, model="nb", R=args.fit_draws)

    print("Fitting book NB latent-class model...")
    fit_lc, lc_status = _fit_with_lc_fallback(
        builder=builder,
        spec=lc_spec,
        fit_draws=args.fit_draws,
    )

    coef_nb, coef_nb_status = _safe_coef_table(
        builder=builder,
        fit_result=fit_nb,
        estimate_col="Estimate_baseline",
    )
    coef_lc, coef_lc_status = _safe_coef_table(
        builder=builder,
        fit_result=fit_lc,
        estimate_col="Estimate_latent_class",
    )

    coef_diff = build_coefficient_difference_table(coef_nb, coef_lc)
    spec_diff = build_spec_difference_table(nb_spec, lc_spec)

    metrics = compare_models(
        {
            "NB baseline (book)": fit_nb,
            "NB latent-class (book)": fit_lc,
        }
    ).reset_index()
    metrics["Status"] = metrics["Model"].map(
        {
            "NB baseline (book)": f"book baseline spec; coef={coef_nb_status}",
            "NB latent-class (book)": f"{lc_status}; coef={coef_lc_status}",
        }
    )

    (output_dir / "book_manual_model_metrics.md").write_text(_to_markdown(metrics), encoding="utf-8")
    (output_dir / "book_manual_model_spec_diff.md").write_text(_to_markdown(spec_diff), encoding="utf-8")
    (output_dir / "book_manual_model_coefficient_diff.md").write_text(_to_markdown(coef_diff), encoding="utf-8")
    (output_dir / "book_manual_model_coefficient_diff.html").write_text(
        coef_diff.to_html(index=False),
        encoding="utf-8",
    )

    slide_text = build_slide_summary(metrics_df=metrics, spec_diff_df=spec_diff)
    (output_dir / "book_manual_model_differences.md").write_text(slide_text, encoding="utf-8")

    print(f"Assets written to: {output_dir.resolve()}")
    print("Created files:")
    print("- book_manual_model_metrics.md")
    print("- book_manual_model_spec_diff.md")
    print("- book_manual_model_coefficient_diff.md")
    print("- book_manual_model_coefficient_diff.html")
    print("- book_manual_model_differences.md")


if __name__ == "__main__":
    main()
