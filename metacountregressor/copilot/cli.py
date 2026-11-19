"""Command-line driver for the experiment copilot.

Usage
-----
Interactive wizard (recommended)::

    python -m metacountregressor.copilot

Non-interactive quickstart::

    python -m metacountregressor.copilot --auto --data crashes.csv ^
        --response FREQ --family count --algo sa --max-iter 200 --R 50 --run

In Python::

    from metacountregressor.copilot import run_cli
    run_cli()          # same as ``python -m metacountregressor.copilot``
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .advisor import OpenAICompatibleAdvisor, RuleBasedAdvisor
from .copilot import ExperimentCopilot
from .profile import load_and_profile
from .schema import Turn, spec_from_dict

BANNER = r"""
  __  __      _        ____            _        _
 |  \/  | ___| |_ __ _/ ___|___  _ __ | |_     / \   ___ _ __
 | |\/| |/ _ \ __/ _` | |   / _ \| '_ \| __|   / _ \ / __| '_ \
 | |  | |  __/ || (_| | |__| (_) | | | | |_   / ___ \ (__| |_) |
 |_|  |_|\___|\__\__,_|\____\___/|_| |_|\__| /_/   \_\___| .__/
                                                        |_|
  experiment copilot for metacountregressor
"""

HELP = """
Answers
  Type a normal answer when a question is shown.  Examples are printed with
  each question.  "none" is valid for optional ID/offset/group columns.

Commands
  /start PATH        load a data source and begin a new setup
  /turn              reprint the current question
  /status            show the current stage and a spec summary
  /spec              print the validated ExperimentSpec as JSON
  /constraints       print the tiered constraint hierarchy
  /run               build the evaluator and run the search
  /diagnose          inspect the last result and propose constraint patches
  /apply             apply the diagnosis patch (without rerunning)
  /rerun             apply the diagnosis patch and run again
  /save PATH         save the ExperimentSpec JSON
  /load PATH         load a previously saved ExperimentSpec JSON
  /help              show this help
  /quit              exit

Typical flow
  /start data/crashes.csv
  FREQ
  count
  ID
  OFFSET
  none
  accept
  accept
  bic
  sa
  1
  300
  50
  /run
  /diagnose
  /apply
  /run
"""


def _stdin_is_tty() -> bool:
    try:
        return bool(sys.stdin and sys.stdin.isatty())
    except Exception:
        return False


def _print_turn(turn: Optional[Turn]) -> None:
    if turn is None:
        return
    print(f"\n? {turn.question}")
    if turn.options:
        print(f"  options : {', '.join(map(str, turn.options[:20]))}")
    if turn.examples:
        print(f"  examples: {' | '.join(map(str, turn.examples[:5]))}")
    if turn.hint:
        print(f"  note    : {turn.hint}")


def _spec_summary(cp: ExperimentCopilot) -> str:
    spec = cp.spec
    if spec is None:
        return "no spec loaded"
    included = spec.included_variables
    return (
        f"family={spec.family} response={spec.response} "
        f"id={spec.id_col} offset={spec.offset_col} "
        f"group={spec.group_id_col} variables={len(included)} "
        f"algo={spec.algo} max_iter={spec.max_iter} R={spec.R}"
    )


def _print_result(result: Dict[str, Any]) -> None:
    print("\n================ SEARCH RESULT ================")
    print(f"  algorithm : {result.get('algorithm')}")
    print(f"  best BIC  : {result.get('best_score')}")
    print(f"  family    : {result.get('family')}")
    print(f"  terms     : {result.get('n_active')}")
    print(f"  fits      : {result.get('evaluations')}")
    if result.get("elapsed") is not None:
        print(f"  elapsed   : {result.get('elapsed'):.1f}s")
    print("===============================================")


def _print_diagnosis(diag: Dict[str, Any]) -> None:
    print("\n================ DIAGNOSIS ====================")
    print(f"  {diag.get('summary')}")
    for issue in diag.get("issues", []):
        print(
            f"  [{issue.get('severity', 'info')}] "
            f"{issue.get('code', '')}: {issue.get('message', '')}"
        )
    print("===============================================")
    if diag.get("next_spec") is not None:
        print("  A patched spec is available.  Use /apply (or /rerun).")


def _clean_path_arg(arg: str) -> str:
    text = arg.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


def _handle_command(cp: ExperimentCopilot, line: str, turn: Optional[Turn]):
    parts = line.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""
    path_arg = _clean_path_arg(arg)

    if cmd in {"/quit", "/exit", "/q"}:
        raise SystemExit(0)
    if cmd in {"/help", "/h", "/?"}:
        print(HELP)
        return turn
    if cmd == "/start":
        if not arg:
            print("usage: /start PATH")
            return turn
        turn = cp.start(os.path.expanduser(path_arg))
        print(f"loaded {cp.profile.n_rows} rows x {cp.profile.n_cols} columns")
        _print_turn(turn)
        return turn
    if cmd == "/turn":
        _print_turn(turn)
        return turn
    if cmd == "/status":
        print(f"stage: {cp.stage}")
        print(f"spec : {_spec_summary(cp)}")
        _print_turn(turn)
        return turn
    if cmd == "/spec":
        print(json.dumps(cp.build_spec().to_dict(), indent=2))
        return turn
    if cmd == "/constraints":
        if cp.spec is None:
            print("no spec loaded")
            return turn
        for tier in cp.spec.constraints:
            print(f"\n[{tier.tier}] {tier.rationale}")
            for rule in tier.rules:
                print(f"  - {rule}")
        return turn
    if cmd == "/run":
        try:
            result = cp.run(verbose=True)
            _print_result(result)
        except Exception as exc:
            print(f"run failed: {type(exc).__name__}: {exc}")
        return turn
    if cmd == "/diagnose":
        try:
            diag = cp.diagnose()
            _print_diagnosis(diag)
        except Exception as exc:
            print(f"diagnose failed: {type(exc).__name__}: {exc}")
        return turn
    if cmd == "/apply":
        try:
            spec = cp.apply_diagnosis()
            print("patched spec applied.")
            print(_spec_summary(cp))
            print(f"constraints: {[(t.tier, len(t.rules)) for t in spec.constraints]}")
        except Exception as exc:
            print(f"apply failed: {type(exc).__name__}: {exc}")
        return turn
    if cmd == "/rerun":
        try:
            cp.apply_diagnosis()
            print("patched spec applied; running...")
            result = cp.run(verbose=True)
            _print_result(result)
        except Exception as exc:
            print(f"rerun failed: {type(exc).__name__}: {exc}")
        return turn
    if cmd == "/save":
        if not arg:
            print("usage: /save PATH")
            return turn
        path = Path(os.path.expanduser(path_arg))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cp.build_spec().to_dict(), indent=2))
        print(f"saved {path}")
        return turn
    if cmd == "/load":
        if not arg:
            print("usage: /load PATH")
            return turn
        data = json.loads(Path(os.path.expanduser(path_arg)).read_text())
        cp.spec = spec_from_dict(data)
        cp.profile = load_and_profile(cp.spec.data_path)
        cp.stage = "done"
        print("loaded spec; use /spec, /run, /diagnose")
        print(_spec_summary(cp))
        return Turn("Setup loaded. Use /run or /diagnose.", "done", done=True)
    print(f"unknown command {cmd!r}; type /help")
    return turn


def _auto_answer(cp: ExperimentCopilot, turn: Turn, args: argparse.Namespace) -> str:
    field = turn.field
    mapping = {
        "response": args.response,
        "family": args.family,
        "id_col": args.id_col,
        "offset_col": args.offset_col,
        "group_id_col": args.group_id_col,
        "aadt_col": args.aadt_col,
        "baseline_vars": args.baseline_vars,
        "local_vars": args.local_vars,
        "variables": args.variables,
        "random_params": args.random_params,
        "constraints": args.constraints,
        "objective": args.objective,
        "algo": args.algo,
        "max_latent_classes": str(args.max_latent_classes),
        "max_iter": str(args.max_iter),
        "R": str(args.R),
    }
    value = mapping.get(field)
    if value:
        return str(value)
    if field in {"variables", "constraints"}:
        return "accept"
    if turn.examples:
        return str(turn.examples[0])
    if field in {"id_col", "offset_col", "group_id_col"}:
        return "none"
    return "accept"


def _run_auto(cp: ExperimentCopilot, args: argparse.Namespace) -> ExperimentCopilot:
    turn = cp.start(args.data)
    guard = 0
    while turn is not None and not turn.done:
        guard += 1
        if guard > 50:
            raise RuntimeError("copilot did not reach a terminal state")
        answer = _auto_answer(cp, turn, args)
        print(f"  {turn.field}: {answer}")
        turn = cp.answer(answer)
    spec = cp.build_spec()
    print("\nValidated ExperimentSpec")
    print(f"  {_spec_summary(cp)}")
    print(f"  variables: {spec.included_variables}")
    print(f"  constraints: {[(t.tier, len(t.rules)) for t in spec.constraints]}")
    if args.run:
        result = cp.run(verbose=False)
        _print_result(result)
        if args.diagnose:
            _print_diagnosis(cp.diagnose())
    return cp


def _interactive(cp: ExperimentCopilot, args: argparse.Namespace) -> int:
    print(BANNER)
    print("Type /help for commands.  Ctrl+C exits.")
    turn: Optional[Turn] = None
    if args.data:
        turn = cp.start(args.data)
        print(f"loaded {cp.profile.n_rows} rows x {cp.profile.n_cols} columns")
        _print_turn(turn)
    elif args.config:
        data = json.loads(Path(os.path.expanduser(args.config)).read_text())
        cp.spec = spec_from_dict(data)
        cp.profile = load_and_profile(cp.spec.data_path)
        cp.stage = "done"
        print(_spec_summary(cp))
    else:
        print("No data loaded.  Start with: /start PATH")
        print("or run: python -m metacountregressor.copilot --auto --data PATH ...")

    while True:
        try:
            line = input("\ncopilot> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return 0
        if not line:
            continue
        if line.startswith("/"):
            turn = _handle_command(cp, line, turn)
            continue
        if turn is None:
            print("Use /start PATH first.")
            continue
        if turn.done:
            print("Setup complete.  Use /spec, /run, /diagnose, or /start PATH.")
            continue
        try:
            turn = cp.answer(line)
        except Exception as exc:
            print(f"invalid answer: {type(exc).__name__}: {exc}")
            _print_turn(turn)
            continue
        _print_turn(turn)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="metacount-copilot",
        description=(
            "LLM/rule-based experiment setup for metacountregressor. "
            "Run with no arguments for the interactive wizard."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=HELP,
    )
    p.add_argument("--data", help="CSV/Parquet/Excel/JSON/Stata data source")
    p.add_argument("--config", help="load an ExperimentSpec JSON and skip the wizard")
    p.add_argument("--auto", action="store_true", help="non-interactive quickstart")
    p.add_argument("--run", action="store_true", help="run the search after setup")
    p.add_argument("--diagnose", action="store_true", help="diagnose after --run")
    p.add_argument("--response", help="response column")
    p.add_argument("--family", default="count", help="count or cmf")
    p.add_argument("--id-col", dest="id_col", default=None, help="ID column")
    p.add_argument("--offset-col", dest="offset_col", default=None, help="offset column")
    p.add_argument("--group-id-col", dest="group_id_col", default=None, help="grouping column")
    p.add_argument("--aadt-col", dest="aadt_col", default=None, help="AADT column (CMF)")
    p.add_argument("--baseline-vars", dest="baseline_vars", default=None, help="CMF baseline variables")
    p.add_argument("--local-vars", dest="local_vars", default=None, help="CMF local variables")
    p.add_argument("--variables", default=None, help="'accept' or comma-separated columns to exclude")
    p.add_argument(
        "--random-params",
        dest="random_params",
        default="accept",
        help=(
            "'accept', 'none', or NAME=dist pairs, e.g. "
            "'CURVES@logAADT=lognormal, WIDTH@logAADT=normal'"
        ),
    )
    p.add_argument("--constraints", default=None, help="'accept' or a constraint edit")
    p.add_argument("--objective", default="bic", help="bic/aic/loglik")
    p.add_argument("--algo", default="sa", help="sa/ga/hs/de/delphos_rl")
    p.add_argument("--max-iter", dest="max_iter", type=int, default=300)
    p.add_argument("--max-latent-classes", dest="max_latent_classes", type=int, default=1)
    p.add_argument("--R", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", dest="output_dir", default="metacount_results")
    p.add_argument("--llm", action="store_true", help="use the OpenAI-compatible LLM advisor")
    p.add_argument(
        "--no-llm",
        action="store_true",
        help="force the deterministic rule-based advisor even if env vars are set",
    )
    p.add_argument("--llm-base-url", dest="llm_base_url", default=None)
    p.add_argument("--llm-model", dest="llm_model", default=None)
    p.add_argument("--llm-api-key", dest="llm_api_key", default=None)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    # The Run/Debug output panes in IDEs are read-only.  Starting the
    # interactive wizard there looks like a hang, so give the user the two
    # useful alternatives instead.
    if not args.auto and not _stdin_is_tty() and os.environ.get(
        "MCR_FORCE_INTERACTIVE"
    ) != "1":
        print(
            "This is the interactive copilot, but stdin is not a terminal. "
            "Run it in the IDE Terminal or a normal PowerShell/Command Prompt."
        )
        print("")
        print("Non-interactive example:")
        print(
            "  python -m metacountregressor.copilot --auto --data PATH "
            "--response FREQ --family count --run --diagnose"
        )
        print("")
        print(
            "Or set MCR_FORCE_INTERACTIVE=1 if your console does support "
            "typed input."
        )
        return 2

    advisor = None
    if args.no_llm:
        advisor = RuleBasedAdvisor()
    elif args.llm or os.environ.get("METACOUNT_LLM_BASE_URL"):
        try:
            advisor = OpenAICompatibleAdvisor(
                base_url=args.llm_base_url or "",
                api_key=args.llm_api_key or "",
                model=args.llm_model or "",
            )
            print(f"LLM advisor: {advisor.model} @ {advisor.base_url}")
        except Exception as exc:
            print(f"LLM advisor unavailable ({exc}); using rule-based advisor")
            advisor = RuleBasedAdvisor()
    else:
        advisor = RuleBasedAdvisor()

    cp = ExperimentCopilot(
        advisor=advisor,
        max_latent_classes=args.max_latent_classes,
        R=args.R,
        max_iter=args.max_iter,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    if args.auto:
        if not args.data:
            print("--auto requires --data")
            return 2
        _run_auto(cp, args)
        return 0
    return _interactive(cp, args)


def run_cli(argv: Optional[list[str]] = None) -> int:
    """Importable alias: ``from metacountregressor.copilot import run_cli; run_cli()``."""
    return main(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

