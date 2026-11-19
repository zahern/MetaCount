"""Offline demo: walk the copilot through Example 16-3 without an LLM.

Run:
    python -m metacountregressor.copilot.demo
    python -m metacountregressor.copilot.demo --run
"""

from __future__ import annotations

import argparse
import os
import tempfile

from metacountregressor import load_example16_3_model_data

from .copilot import ExperimentCopilot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="run a tiny search")
    parser.add_argument("--max-iter", type=int, default=5)
    parser.add_argument("--R", type=int, default=10)
    args = parser.parse_args()

    df = load_example16_3_model_data()
    path = os.path.join(tempfile.gettempdir(), "metacount_delphos_demo.csv")
    df.to_csv(path, index=False)

    cp = ExperimentCopilot(max_iter=args.max_iter, R=args.R)
    turn = cp.start(path)
    answers = [
        "FREQ",      # response
        "count",     # family
        "ID",        # id
        "OFFSET",    # offset
        "none",      # group
        "accept",    # variable catalogue
        "accept",    # constraint hierarchy
        "bic",       # objective
        "sa",        # search driver
        "1",         # max latent classes
        str(args.max_iter),
        str(args.R),
    ]
    print(f"Q: {turn.question}\n   options={turn.options[:8]}")
    for answer in answers:
        turn = cp.answer(answer)
        print(f"> {answer}")
        print(f"Q: {turn.question}")
        if turn.examples:
            print(f"   examples={turn.examples[:3]}")

    spec = cp.build_spec()
    print("\nValidated ExperimentSpec")
    print(f"  data      : {spec.data_path}")
    print(f"  response  : {spec.response}")
    print(f"  family    : {spec.family}")
    print(f"  id/offset : {spec.id_col} / {spec.offset_col}")
    print(f"  variables : {spec.included_variables}")
    print(f"  constraints: {[(t.tier, len(t.rules)) for t in spec.constraints]}")
    print(f"  search    : {spec.algo}, max_iter={spec.max_iter}, R={spec.R}")

    if args.run:
        result = cp.run()
        print("\nSearch result")
        print(f"  best BIC : {result['best_score']:.3f}")
        print(f"  family   : {result['family']}")
        print(f"  terms    : {result['n_active']}")
        diagnosis = cp.diagnose()
        print("\nDiagnosis")
        print(f"  {diagnosis['summary']}")
        for issue in diagnosis["issues"]:
            print(f"  - [{issue['severity']}] {issue['code']}: {issue['message']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


