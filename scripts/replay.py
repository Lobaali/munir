"""Measure Module 6 optimisation steps in the required order.

Lab 6's lesson is "meter first, optimise second". This script therefore
runs the SAME small replay through four profiles:

1. before      = old prompt, no response cache, no cascade
2. prompt      = stable-prefix prompt, no response cache, no cascade
3. cache       = stable prompt + exact response cache, no cascade
4. cascade     = stable prompt + response cache + cheap-first cascade

For every profile we record:
- measured model spend from CostMeter
- cached input share reported by the model boundary
- p50 turn latency
- the existing Section 4 golden-set pass rate

The final table puts the evaluation verdict beside the cost change. A cheaper
configuration that fails the golden set is not an optimisation we should ship.

Example:
    python scripts/replay.py --limit 40
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "eval" / "golden" / "regression_set.yaml"
OUT = ROOT / "eval" / "out"

# Keep this list small enough for a quick lab run. The full golden set is still
# used by the Section 4 harness; this replay measures economics on representative
# repeated turns, not a replacement for the golden set.
REPLAY_QUESTIONS = [
    "How much does a transcript cost?",
    "What documents do I need for a transcript?",
    "What is the deadline for dropping a course?",
    "How much does a transcript cost?",
    "How do I book an advisor appointment?",
    "What documents do I need for a transcript?",
    "When is the add/drop deadline?",
    "How much does a transcript cost?",
    "Can I get a transcript by mail?",
    "What documents do I need for a transcript?",
]

PROFILES = {
    "before": {
        "MUNIR_FAQ_PROMPT": "answer_faq.v1",
        "MUNIR_CACHE_ENABLED": "false",
        "MUNIR_CASCADE_ENABLED": "false",
    },
    "prompt": {
        "MUNIR_FAQ_PROMPT": "answer_faq.v2",
        "MUNIR_CACHE_ENABLED": "false",
        "MUNIR_CASCADE_ENABLED": "false",
    },
    "cache": {
        "MUNIR_FAQ_PROMPT": "answer_faq.v2",
        "MUNIR_CACHE_ENABLED": "true",
        "MUNIR_CASCADE_ENABLED": "false",
    },
    "cascade": {
        "MUNIR_FAQ_PROMPT": "answer_faq.v2",
        "MUNIR_CACHE_ENABLED": "true",
        "MUNIR_CASCADE_ENABLED": "true",
    },
}


def run_economics(profile: str, questions: list[str]) -> dict:
    """Run real Munir turns and read the real CostMeter."""

    for key, value in PROFILES[profile].items():
        os.environ[key] = value

    from munir.app import build_assistant
    from munir.domain.session import Session

    assistant = build_assistant()
    latencies = []

    for question in questions:
        started = time.perf_counter()
        assistant.ask(
            question,
            Session(student_id="STU-100001"),
            remember=False,
        )
        latencies.append(
            (time.perf_counter() - started) * 1000
        )

    latencies.sort()
    p50 = latencies[(len(latencies) - 1) // 2]

    return {
        "profile": profile,
        "questions": len(questions),
        "model_calls": len(assistant.meter.records),
        "cost_sar": assistant.meter.total_sar,
        "cached_input_share": assistant.meter.cached_input_share(),
        "p50_latency_ms": round(p50, 1),
        "response_cache": (
            assistant.cache.stats.__dict__
            if assistant.cache is not None
            else None
        ),
        "spend_by_stage": assistant.meter.by("stage"),
    }


def run_eval(profile: str, limit: int | None) -> dict:
    """Run the Section 4 harness under the same optimisation profile."""

    env = os.environ.copy()
    env.update(PROFILES[profile])

    command = [
        sys.executable,
        str(ROOT / "eval" / "harness.py"),
        "--label",
        f"module6_{profile}",
    ]

    if limit is not None:
        command.extend(["--limit", str(limit)])

    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    if completed.returncode not in {0, 1}:
        raise RuntimeError(
            f"evaluation harness failed for {profile}:\n"
            f"{completed.stdout}\n{completed.stderr}"
        )

    report_path = OUT / f"eval_module6_{profile}.json"
    return json.loads(
        report_path.read_text(encoding="utf-8")
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=40,
        help="golden-set cases used for the evaluation verdict",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="write the combined before/after report",
    )
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    rows = []

    for profile in PROFILES:
        economics = run_economics(
            profile,
            REPLAY_QUESTIONS,
        )
        evaluation = run_eval(
            profile,
            args.limit,
        )

        rows.append(
            {
                **economics,
                "eval_cases": evaluation["cases"],
                "eval_pass_rate": evaluation["pass_rate"],
                "eval_safety_rate": evaluation["slices"].get("risk", {}).get("safety"),
            }
        )

    print("\nModule 6 optimisation replay")
    print("=" * 110)
    print(
        f"{'step':<10} {'cost SAR':>12} {'cache input':>13} "
        f"{'p50 ms':>10} {'model calls':>12} {'eval':>10} {'safety':>10}"
    )
    print("-" * 110)

    for row in rows:
        print(
            f"{row['profile']:<10} "
            f"{row['cost_sar']:>12.6f} "
            f"{row['cached_input_share']:>12.1%} "
            f"{row['p50_latency_ms']:>10.1f} "
            f"{row['model_calls']:>12} "
            f"{row['eval_pass_rate']:>9.1%} "
            f"{(row['eval_safety_rate'] if row['eval_safety_rate'] is not None else 0):>9.1%}"
        )

    print("\nThe evaluation columns are the release verdict, not decoration.")
    print("If cost falls but the safety slice falls below 100%, do not ship the step.")

    if args.write:
        output = OUT / "cost_optimization_comparison.json"
        output.write_text(
            json.dumps(
                {"profiles": rows},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwritten: {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
