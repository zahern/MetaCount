"""
Fit a latent-class count model using DE warm-start and print full results.

Usage:
    python test_latent_class_de_warmup_fit.py
    python test_latent_class_de_warmup_fit.py --R 120 --de-maxiter 20 --de-popsize 10
"""

from __future__ import annotations

import argparse
import pprint
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metacountregressor import (
    ExperimentBuilder,
    load_book_latent_class_spec,
    load_example16_3_model_data,
)


def _print_title(title: str) -> None:
    print("\n" + "=" * 88)
    print(f"  {title}")
    print("=" * 88)


def _print_de_report(report: dict) -> None:
    _print_title("DE WARM-START REPORT")
    pprint.pprint(report, width=120, sort_dicts=False)

    single = report.get("single_class") or {}
    if single:
        start_obj = single.get("start_obj")
        de_obj = single.get("de_obj")
        final_obj = single.get("final_obj")
        start_ll = -float(start_obj) if start_obj is not None else None
        de_ll = -float(de_obj) if de_obj is not None else None
        final_ll = -float(final_obj) if final_obj is not None else None
        print("\nSingle-class warm-start LL diagnostics:")
        print(f"  start LL: {start_ll}")
        print(f"  warmup LL: {de_ll}")
        print(f"  final single-class LL: {final_ll}")


def _print_summary_table(summary: dict) -> None:
    _print_title("MODEL FIT SUMMARY")
    metric_rows = []
    for key in ["loglik", "aic", "bic", "num_parm", "n_obs", "latent_classes"]:
        if key in summary:
            metric_rows.append({"Metric": key, "Value": summary[key]})

    if metric_rows:
        summary_df = pd.DataFrame(metric_rows)
        print(summary_df.to_string(index=False))

    class_probs = summary.get("class_probs")
    if class_probs:
        probs_df = pd.DataFrame(
            {
                "Class": [f"Class {i + 1}" for i in range(len(class_probs))],
                "Probability": class_probs,
            }
        )
        print("\nMarginal class probabilities:")
        print(probs_df.to_string(index=False))


def _print_coef_table(builder: ExperimentBuilder, fit: dict) -> None:
    _print_title("FULL COEFFICIENT TABLE")
    coef_df = builder.print_coefficients(fit).copy()
    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        200,
        "display.max_colwidth",
        None,
    ):
        print(coef_df.to_string(index=False))


def _print_predictions_preview(fit: dict) -> None:
    _print_title("PREDICTION SNAPSHOT")
    y = np.asarray(fit["data"]["y"]).reshape(-1)
    pred = np.asarray(fit["predictions"]).reshape(-1)

    preview = pd.DataFrame(
        {
            "Observed": y[:20],
            "Predicted": pred[:20],
            "Residual": y[:20] - pred[:20],
        }
    )
    print(preview.to_string(index=False))

    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    mae = float(np.mean(np.abs(y - pred)))
    print("\nAggregate prediction diagnostics:")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE : {mae:.6f}")


def _evaluate_latent_vs_actual_class(
    builder: ExperimentBuilder,
    fit: dict,
    actual_class_col: str,
    *,
    print_details: bool = True,
) -> dict:
    if print_details:
        _print_title("LATENT-CLASS DISCOVERY CHECK VS ACTUAL FUNCTIONAL CLASS")

    probs = builder.compute_latent_class_probabilities(
        fit_result=fit,
        true_class_col=actual_class_col,
    )

    class_prob_cols = [c for c in probs.columns if c.startswith("class_") and c.endswith("_prob")]
    if not class_prob_cols:
        if print_details:
            print("No latent-class probability columns were found.")
        return {
            "n_observations": 0,
            "discovered_classes": 0,
            "actual_classes": 0,
            "purity": np.nan,
            "majority_map_accuracy": np.nan,
            "chi2": np.nan,
            "p_value": np.nan,
            "cramers_v": np.nan,
            "collapsed": True,
            "hard_collapsed": True,
            "min_posterior_share": np.nan,
            "min_hard_share": np.nan,
            "mean_posteriors": {},
            "hard_shares": {},
        }

    prob_mat = probs[class_prob_cols].to_numpy(dtype=float)
    probs["pred_latent_class"] = np.argmax(prob_mat, axis=1) + 1
    mean_post = probs[class_prob_cols].mean(axis=0)
    min_posterior_share = float(mean_post.min()) if len(mean_post) else float("nan")
    hard_share = probs["pred_latent_class"].value_counts(normalize=True).sort_index()
    min_hard_share = float(hard_share.min()) if len(hard_share) else float("nan")

    if print_details:
        print("Mean posterior probability by latent class:")
        for col, val in mean_post.items():
            print(f"  {col}: {float(val):.6f}")
        print("Hard-assignment proportion by latent class:")
        for cls, val in hard_share.items():
            print(f"  class_{int(cls)}: {float(val):.6f}")

    if actual_class_col == "FC_LABEL":
        actual_labels = probs[actual_class_col].astype(str)
    else:
        actual_labels = probs[actual_class_col].astype("Int64").astype(str)

    contingency = pd.crosstab(
        probs["pred_latent_class"],
        actual_labels,
        rownames=["Predicted latent class"],
        colnames=[f"Actual {actual_class_col}"],
    )

    requested_classes = int(fit["spec"].latent_classes)
    discovered_classes = int(contingency.shape[0])
    collapsed = discovered_classes < requested_classes
    if discovered_classes < requested_classes:
        if print_details:
            print(
                f"\nWARNING: Hard class assignment collapsed to {discovered_classes} class(es) "
                f"while the model was fit with latent_classes={requested_classes}."
            )

    if print_details:
        print("Contingency table (latent class x actual class):")
        print(contingency.to_string())

    row_props = contingency.div(contingency.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    if print_details:
        print("\nRow-normalized composition (how each latent class maps to actual classes):")
        print(row_props.round(4).to_string())

    n = int(contingency.to_numpy().sum())
    if n == 0:
        if print_details:
            print("No observations available for class-comparison diagnostics.")
        return {
            "n_observations": 0,
            "discovered_classes": discovered_classes,
            "actual_classes": int(contingency.shape[1]),
            "purity": np.nan,
            "majority_map_accuracy": np.nan,
            "chi2": np.nan,
            "p_value": np.nan,
            "cramers_v": np.nan,
            "collapsed": collapsed,
            "hard_collapsed": collapsed,
            "min_posterior_share": min_posterior_share,
            "min_hard_share": min_hard_share,
            "mean_posteriors": {k: float(v) for k, v in mean_post.to_dict().items()},
            "hard_shares": {f"class_{int(k)}": float(v) for k, v in hard_share.to_dict().items()},
        }

    purity = float(contingency.max(axis=1).sum() / n)

    mapped_counts = 0
    for cls in contingency.index:
        row = contingency.loc[cls]
        mapped_counts += int(row.max())
    majority_map_accuracy = float(mapped_counts / n)

    chi2, p_value, dof, _ = chi2_contingency(contingency)
    r, k = contingency.shape
    denom = min(max(r - 1, 0), max(k - 1, 0))
    cramers_v = float(np.sqrt((chi2 / n) / denom)) if denom > 0 else 0.0

    if print_details:
        print("\nDiscovery diagnostics:")
        print(f"  n observations             : {n}")
        print(f"  latent classes (predicted): {r}")
        print(f"  actual classes            : {k}")
        print(f"  Purity                    : {purity:.6f}")
        print(f"  Majority-map accuracy     : {majority_map_accuracy:.6f}")
        print(f"  Chi-square                : {chi2:.6f}")
        print(f"  p-value                   : {p_value:.6g}")
        print(f"  Cramer's V                : {cramers_v:.6f}")

        print("\nBest matching actual class per latent class:")
        for cls in contingency.index:
            row = contingency.loc[cls]
            winner = row.idxmax()
            winner_n = int(row.max())
            cls_total = int(row.sum())
            share = (winner_n / cls_total) if cls_total > 0 else 0.0
            print(
                f"  Latent class {int(cls)} -> {winner} "
                f"({winner_n}/{cls_total}, share={share:.4f})"
            )

    return {
        "n_observations": n,
        "discovered_classes": r,
        "actual_classes": k,
        "purity": purity,
        "majority_map_accuracy": majority_map_accuracy,
        "chi2": float(chi2),
        "p_value": float(p_value),
        "cramers_v": cramers_v,
        "collapsed": collapsed,
        "hard_collapsed": collapsed,
        "min_posterior_share": min_posterior_share,
        "min_hard_share": min_hard_share,
        "mean_posteriors": {k: float(v) for k, v in mean_post.to_dict().items()},
        "hard_shares": {f"class_{int(k)}": float(v) for k, v in hard_share.to_dict().items()},
    }


def _fit_latent_model(
    builder: ExperimentBuilder,
    manual_spec: dict,
    args: argparse.Namespace,
    *,
    print_report: bool,
    de_seed: int | None = None,
) -> dict:
    if de_seed is None:
        de_seed = int(args.de_seed)

    return builder.fit_manual_model(
        manual_spec=manual_spec,
        model="nb",
        R=args.R,
        print_report=print_report,
        use_prefit_start=True,
        continuous_de_warm_start=True,
        de_maxiter=args.de_maxiter,
        de_popsize=args.de_popsize,
        de_rel_span=args.de_rel_span,
        de_abs_span=args.de_abs_span,
        de_seed=int(de_seed),
        latent_fast_mode=bool(args.latent_fast_mode),
        latent_random_start=False,
    )


def _fit_with_anti_collapse(
    df: pd.DataFrame,
    manual_spec: dict,
    args: argparse.Namespace,
    actual_class_col: str,
) -> tuple[dict, dict, dict]:
    """Try multiple seeds and pick the best non-collapsed fit when available."""

    requested_classes = int(manual_spec.get("latent_classes", 1))
    retries = max(1, int(args.anti_collapse_restarts))
    seed_step = int(args.anti_collapse_seed_step)
    min_share = float(args.min_class_share)
    min_hard_share_required = float(args.min_hard_class_share)

    best_any = None
    best_noncollapsed = None
    trial_rows = []

    for i in range(retries):
        seed_i = int(args.de_seed) + i * seed_step
        builder_i = ExperimentBuilder(
            df=df,
            id_col="ID",
            y_col="FREQ",
            offset_col="OFFSET",
        )
        fit_i = _fit_latent_model(builder_i, manual_spec, args, print_report=False, de_seed=seed_i)
        eval_i = _evaluate_latent_vs_actual_class(
            builder_i,
            fit_i,
            actual_class_col=actual_class_col,
            print_details=False,
        )

        summary_i = fit_i.get("summary", {}) or {}
        loglik_i = float(summary_i.get("loglik", -np.inf))
        discovered_i = int(eval_i.get("discovered_classes", 0))
        min_post_i = float(eval_i.get("min_posterior_share", np.nan))
        min_hard_i = float(eval_i.get("min_hard_share", np.nan))

        is_hard_noncollapsed = discovered_i >= requested_classes
        is_soft_noncollapsed = np.isfinite(min_post_i) and (min_post_i >= min_share)
        is_hard_share_ok = np.isfinite(min_hard_i) and (min_hard_i >= min_hard_share_required)
        is_noncollapsed = is_hard_noncollapsed and is_soft_noncollapsed and is_hard_share_ok

        trial_rows.append(
            {
                "trial": i,
                "seed": seed_i,
                "loglik": loglik_i,
                "discovered_classes": discovered_i,
                "min_posterior_share": min_post_i,
                "min_hard_share": min_hard_i,
                "hard_noncollapsed": is_hard_noncollapsed,
                "soft_noncollapsed": is_soft_noncollapsed,
                "hard_share_ok": is_hard_share_ok,
                "selected_pool": "noncollapsed" if is_noncollapsed else "collapsed",
            }
        )

        rec = {"fit": fit_i, "eval": eval_i, "seed": seed_i, "loglik": loglik_i}
        if (best_any is None) or (loglik_i > best_any["loglik"]):
            best_any = rec
        if is_noncollapsed and ((best_noncollapsed is None) or (loglik_i > best_noncollapsed["loglik"])):
            best_noncollapsed = rec

    if best_noncollapsed is None and not bool(args.allow_imbalanced_fallback):
        trials_df = pd.DataFrame(trial_rows)
        raise RuntimeError(
            "No balanced latent-class solution found across seed restarts. "
            "Increase --anti-collapse-restarts, relax --min-class-share/--min-hard-class-share, "
            "or set --allow-imbalanced-fallback to proceed anyway.\n\n"
            f"Trial diagnostics:\n{trials_df.to_string(index=False)}"
        )

    selected = best_noncollapsed if best_noncollapsed is not None else best_any
    selected_seed = int(selected["seed"])

    builder_final = ExperimentBuilder(
        df=df,
        id_col="ID",
        y_col="FREQ",
        offset_col="OFFSET",
    )
    fit_final = _fit_latent_model(builder_final, manual_spec, args, print_report=True, de_seed=selected_seed)
    eval_final = _evaluate_latent_vs_actual_class(
        builder_final,
        fit_final,
        actual_class_col=actual_class_col,
        print_details=False,
    )

    selection_meta = {
        "selected_seed": selected_seed,
        "requested_classes": requested_classes,
        "min_class_share": min_share,
        "min_hard_class_share": min_hard_share_required,
        "trials": trial_rows,
        "used_noncollapsed_pool": best_noncollapsed is not None,
        "balance_constraints_satisfied": best_noncollapsed is not None,
        "allow_imbalanced_fallback": bool(args.allow_imbalanced_fallback),
    }
    return fit_final, eval_final, selection_meta


def _run_side_by_side_comparison(df: pd.DataFrame, args: argparse.Namespace, actual_class_col: str) -> None:
    _print_title("SIDE-BY-SIDE COMPARISON (BLIND VS FC-MEMBERSHIP)")
    print("Running two fits with identical settings except membership design.")

    rows = []
    modes = [
        ("blind_no_fc_membership", []),
        ("fc_membership", ["FC_ENCODED"]),
    ]

    for mode_name, membership_terms in modes:
        print(f"\n--- Fitting mode: {mode_name} ---")
        spec_mode = dict(load_book_latent_class_spec())
        spec_mode["membership_terms"] = membership_terms
        fit, compare, selection = _fit_with_anti_collapse(
            df=df,
            manual_spec=spec_mode,
            args=args,
            actual_class_col=actual_class_col,
        )
        builder_mode = ExperimentBuilder(
            df=df,
            id_col="ID",
            y_col="FREQ",
            offset_col="OFFSET",
        )

        _print_title(f"MODEL DETAILS: {mode_name}")
        _print_de_report(fit.get("de_warm_start_report", {}) or {})
        _print_summary_table(fit.get("summary", {}) or {})
        _print_coef_table(builder_mode, fit)
        _evaluate_latent_vs_actual_class(
            builder_mode,
            fit,
            actual_class_col=actual_class_col,
            print_details=True,
        )

        summary = fit.get("summary", {}) or {}

        single = (fit.get("de_warm_start_report", {}) or {}).get("single_class", {}) or {}
        rows.append(
            {
                "Mode": mode_name,
                "MembershipTerms": str(spec_mode.get("membership_terms", [])),
                "LogLik": summary.get("loglik", np.nan),
                "AIC": summary.get("aic", np.nan),
                "BIC": summary.get("bic", np.nan),
                "DiscoveredClasses": compare.get("discovered_classes", np.nan),
                "Collapsed": bool(compare.get("collapsed", False)),
                "MinPosteriorShare": compare.get("min_posterior_share", np.nan),
                "MinHardShare": compare.get("min_hard_share", np.nan),
                "Purity": compare.get("purity", np.nan),
                "MajorityMapAcc": compare.get("majority_map_accuracy", np.nan),
                "CramersV": compare.get("cramers_v", np.nan),
                "pValue": compare.get("p_value", np.nan),
                "DEAccepted": single.get("accepted", None),
                "DEDeltaObj": single.get("delta_obj", np.nan),
                "SelectedSeed": selection.get("selected_seed", np.nan),
                "UsedNonCollapsedPool": bool(selection.get("used_noncollapsed_pool", False)),
                "BalanceSatisfied": bool(selection.get("balance_constraints_satisfied", False)),
            }
        )

    out = pd.DataFrame(rows)
    _print_title("COMPARISON TABLE")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(out.to_string(index=False))


def _pick_actual_class_col(df: pd.DataFrame) -> str:
    for col in ["FC_LABEL", "FC", "FC_ENCODED"]:
        if col in df.columns:
            return col
    raise ValueError(
        "Could not find a functional-class column. Expected one of: FC_LABEL, FC, FC_ENCODED."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run latent-class NB fit with DE warm-start and print full tables."
    )
    parser.add_argument("--R", type=int, default=120, help="Halton draws for estimation")
    parser.add_argument("--de-maxiter", type=int, default=16, help="DE warm-start max iterations")
    parser.add_argument("--de-popsize", type=int, default=8, help="DE warm-start population size")
    parser.add_argument("--de-rel-span", type=float, default=2.0, help="Relative DE search span")
    parser.add_argument("--de-abs-span", type=float, default=1.0, help="Absolute DE search span")
    parser.add_argument("--de-seed", type=int, default=11, help="DE warm-start RNG seed")
    parser.add_argument(
        "--actual-class-col",
        default=None,
        help="Actual class column for latent-vs-actual comparison (default auto-detect: FC_LABEL/FC/FC_ENCODED).",
    )
    parser.add_argument(
        "--use-fc-membership",
        action="store_true",
        help="If set, keep FC membership terms in the latent model. Default is blind mode (no FC membership terms).",
    )
    parser.add_argument(
        "--compare-blind-vs-fc-membership",
        action="store_true",
        help="Run both blind and FC-membership fits and print a compact side-by-side comparison table.",
    )
    parser.add_argument(
        "--anti-collapse-restarts",
        type=int,
        default=6,
        help="Number of seed restarts used to search for a non-collapsed latent solution.",
    )
    parser.add_argument(
        "--anti-collapse-seed-step",
        type=int,
        default=101,
        help="Seed increment between anti-collapse restarts.",
    )
    parser.add_argument(
        "--min-class-share",
        type=float,
        default=0.10,
        help="Minimum average posterior class share required per class for non-collapsed selection.",
    )
    parser.add_argument(
        "--min-hard-class-share",
        type=float,
        default=0.10,
        help="Minimum hard-assignment class share required per class for non-collapsed selection.",
    )
    parser.add_argument(
        "--allow-imbalanced-fallback",
        action="store_true",
        help="If set, allow falling back to best-loglik fit even when class-balance thresholds are not satisfied.",
    )
    parser.add_argument(
        "--latent-fast-mode",
        action="store_true",
        help="Use latent fast mode for quicker approximate fitting.",
    )
    args = parser.parse_args()

    _print_title("LOAD DATA + SPEC")
    df = load_example16_3_model_data()
    manual_spec = load_book_latent_class_spec()
    actual_class_col = args.actual_class_col or _pick_actual_class_col(df)

    if not args.use_fc_membership:
        manual_spec = dict(manual_spec)
        manual_spec["membership_terms"] = []

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Latent classes requested: {manual_spec.get('latent_classes')}")
    print(f"Actual class column: {actual_class_col}")
    print(f"Blind latent discovery mode: {not args.use_fc_membership}")
    print(f"Membership terms used: {manual_spec.get('membership_terms', [])}")
    print(f"Min posterior class share required: {args.min_class_share:.4f}")
    print(f"Min hard-assignment class share required: {args.min_hard_class_share:.4f}")

    if args.compare_blind_vs_fc_membership:
        _run_side_by_side_comparison(df, args, actual_class_col=actual_class_col)
        _print_title("DONE")
        print("Side-by-side blind vs FC-membership comparison completed successfully.")
        return

    _print_title("FIT LATENT-CLASS COUNT MODEL (DE WARM-START ENABLED)")
    fit, eval_summary, selection_meta = _fit_with_anti_collapse(
        df=df,
        manual_spec=manual_spec,
        args=args,
        actual_class_col=actual_class_col,
    )

    _print_title("ANTI-COLLAPSE SEED SELECTION")
    print(f"Selected seed: {selection_meta['selected_seed']}")
    print(f"Used non-collapsed pool: {selection_meta['used_noncollapsed_pool']}")
    print(f"Balance constraints satisfied: {selection_meta['balance_constraints_satisfied']}")
    print(f"Allow imbalanced fallback: {selection_meta['allow_imbalanced_fallback']}")
    print(
        f"Criteria: discovered_classes >= {selection_meta['requested_classes']} and "
        f"min_posterior_share >= {selection_meta['min_class_share']:.4f} and "
        f"min_hard_share >= {selection_meta['min_hard_class_share']:.4f}"
    )
    trials_df = pd.DataFrame(selection_meta["trials"])
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(trials_df.to_string(index=False))

    builder = ExperimentBuilder(
        df=df,
        id_col="ID",
        y_col="FREQ",
        offset_col="OFFSET",
    )

    _print_de_report(fit.get("de_warm_start_report", {}) or {})
    _print_summary_table(fit.get("summary", {}) or {})
    _print_coef_table(builder, fit)
    _evaluate_latent_vs_actual_class(builder, fit, actual_class_col=actual_class_col, print_details=True)
    _print_predictions_preview(fit)

    _print_title("DONE")
    print("Latent-class DE warm-start fit completed successfully.")


if __name__ == "__main__":
    main()
