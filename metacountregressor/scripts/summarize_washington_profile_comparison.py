from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


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


def _read_required_csv(path: Path, name: str) -> pd.DataFrame:
    file_path = path / name
    if not file_path.exists():
        raise FileNotFoundError(f"Missing required file: {file_path}")
    return pd.read_csv(file_path)


def _split_metrics(metrics_df: pd.DataFrame, split_name: str) -> dict[str, float]:
    sub = metrics_df.loc[metrics_df["Split"] == split_name]
    if sub.empty:
        return {}
    row = sub.iloc[0].to_dict()
    out: dict[str, float] = {}
    for key, value in row.items():
        if key == "Split":
            continue
        try:
            out[key] = float(value)
        except Exception:
            pass
    return out


def _safe_pct_delta(base: float, candidate: float) -> float:
    if not np.isfinite(base) or abs(base) < 1e-12:
        return float("nan")
    return 100.0 * (candidate - base) / abs(base)


def _top_structure(search_df: pd.DataFrame) -> tuple[str, str]:
    if search_df.empty:
        return "(none)", "(none)"
    row = search_df.iloc[0]
    upper = str(row.get("Upper Vars", "(none)"))
    lower = str(row.get("Lower Vars", "(none)"))
    return upper, lower


def _build_comparison(
    core_metrics: pd.DataFrame,
    expanded_metrics: pd.DataFrame,
    core_search: pd.DataFrame,
    expanded_search: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    val_split = "validation (selection fit)"
    test_split = "test (held out)"

    core_val = _split_metrics(core_metrics, val_split)
    exp_val = _split_metrics(expanded_metrics, val_split)
    core_test = _split_metrics(core_metrics, test_split)
    exp_test = _split_metrics(expanded_metrics, test_split)

    key_metrics = ["rmse", "mae", "bias", "corr", "r2", "poisson_dev"]
    rows = []
    for split_name, left, right in [
        ("validation", core_val, exp_val),
        ("test", core_test, exp_test),
    ]:
        for metric in key_metrics:
            base = left.get(metric, float("nan"))
            cand = right.get(metric, float("nan"))
            rows.append(
                {
                    "Split": split_name,
                    "Metric": metric,
                    "Core": base,
                    "Expanded": cand,
                    "Expanded - Core": cand - base,
                    "Pct Delta vs Core": _safe_pct_delta(base, cand),
                }
            )

    comp_df = pd.DataFrame(rows)

    core_upper, core_lower = _top_structure(core_search)
    exp_upper, exp_lower = _top_structure(expanded_search)

    structure_df = pd.DataFrame(
        [
            {
                "Profile": "core",
                "Top upper vars": core_upper,
                "Top lower vars": core_lower,
            },
            {
                "Profile": "expanded",
                "Top upper vars": exp_upper,
                "Top lower vars": exp_lower,
            },
        ]
    )

    return comp_df, structure_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create side-by-side comparison outputs for Washington core vs expanded CMF profile runs."
    )
    parser.add_argument("--core-dir", default="results/washington_cmf", help="Directory for core profile outputs.")
    parser.add_argument("--expanded-dir", default="results/washington_cmf_expanded", help="Directory for expanded profile outputs.")
    parser.add_argument("--output-dir", default="results/washington_cmf_compare", help="Directory for comparison outputs.")

    args = parser.parse_args()

    core_dir = Path(args.core_dir)
    expanded_dir = Path(args.expanded_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    core_metrics = _read_required_csv(core_dir, "validation_metrics.csv")
    expanded_metrics = _read_required_csv(expanded_dir, "validation_metrics.csv")
    core_search = _read_required_csv(core_dir, "search_history_top25.csv")
    expanded_search = _read_required_csv(expanded_dir, "search_history_top25.csv")

    comp_df, structure_df = _build_comparison(core_metrics, expanded_metrics, core_search, expanded_search)

    # Pick a simple recommendation based on test Poisson deviance, then test RMSE.
    core_test = _split_metrics(core_metrics, "test (held out)")
    exp_test = _split_metrics(expanded_metrics, "test (held out)")

    core_pd = core_test.get("poisson_dev", float("nan"))
    exp_pd = exp_test.get("poisson_dev", float("nan"))
    core_rmse = core_test.get("rmse", float("nan"))
    exp_rmse = exp_test.get("rmse", float("nan"))

    if np.isfinite(core_pd) and np.isfinite(exp_pd) and exp_pd < core_pd:
        winner = "expanded"
        reason = f"lower test Poisson deviance ({exp_pd:.4f} < {core_pd:.4f})"
    elif np.isfinite(core_pd) and np.isfinite(exp_pd) and exp_pd > core_pd:
        winner = "core"
        reason = f"lower test Poisson deviance ({core_pd:.4f} < {exp_pd:.4f})"
    elif np.isfinite(core_rmse) and np.isfinite(exp_rmse) and exp_rmse < core_rmse:
        winner = "expanded"
        reason = f"lower test RMSE ({exp_rmse:.4f} < {core_rmse:.4f})"
    elif np.isfinite(core_rmse) and np.isfinite(exp_rmse) and exp_rmse > core_rmse:
        winner = "core"
        reason = f"lower test RMSE ({core_rmse:.4f} < {exp_rmse:.4f})"
    else:
        winner = "tie"
        reason = "insufficient finite test metrics to differentiate"

    (output_dir / "washington_core_vs_expanded_metrics.csv").write_text(comp_df.to_csv(index=False), encoding="utf-8")
    (output_dir / "washington_core_vs_expanded_top_structures.csv").write_text(structure_df.to_csv(index=False), encoding="utf-8")

    md = [
        "Washington Core vs Expanded Profile Comparison",
        "",
        f"Recommendation: {winner} ({reason})",
        "",
        "Top selected structures",
        _to_markdown(structure_df),
        "",
        "Metric deltas (expanded relative to core)",
        _to_markdown(comp_df),
    ]
    (output_dir / "washington_core_vs_expanded_summary.md").write_text("\n".join(md), encoding="utf-8")

    print(f"Comparison outputs written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
