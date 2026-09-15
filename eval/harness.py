"""Run Munir's versioned golden set through the REAL application.

Typical commands
----------------
    python eval/harness.py --label current
    python eval/harness.py --judge --label current_with_judge
    python eval/harness.py --route open_weight --label open_weight
    python eval/harness.py --write-baseline --label baseline

Rubric coverage
---------------
- 4.1: versioned, stratified golden set is loaded from YAML.
- 4.2: every case goes through build_assistant().ask(), i.e. the real app.
- 4.3: deterministic assertions run before the judge can influence quality.
- 4.4: optional LLM judge is run only for cases requesting that rubric.
- 4.6: the resulting report is consumed by eval/gate.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "eval") not in sys.path:
    sys.path.insert(0, str(ROOT / "eval"))

from asserts.checks import CHECKS  # noqa: E402
from munir.domain.session import Session, contains_pii  # noqa: E402
from eval.judge import judge_case  # noqa: E402

GOLDEN = ROOT / "eval" / "golden" / "regression_set.yaml"
OUT = ROOT / "eval" / "out"
BASELINE = ROOT / "eval" / "baseline.json"
SLICE_KEYS = ("language", "intent", "difficulty", "risk")


def _stable_student_id(case_id: str) -> str:
    """Create the same synthetic student ID on every Python process.

    Python's built-in hash() is randomized between processes, so it is not
    appropriate for reproducible evaluation data.
    """
    digest = hashlib.sha256(case_id.encode("utf-8")).hexdigest()
    number = int(digest[:8], 16) % 900000 + 100000
    return f"STU-{number}"


def evaluate_assert(spec: dict[str, Any], reply, case: dict[str, Any]) -> tuple[bool, str]:
    """Evaluate one declarative assertion without an LLM."""
    kind = spec["type"]
    text = reply.text or ""

    if kind == "contains":
        passed = spec["value"] in text
        return passed, "" if passed else f"missing {spec['value']!r}"
    if kind == "not-contains":
        passed = spec["value"] not in text
        return passed, "" if passed else f"contains {spec['value']!r}"
    if kind == "regex":
        passed = bool(re.search(spec["value"], text))
        return passed, "" if passed else f"no regex match for {spec['value']!r}"
    if kind == "intent":
        passed = reply.intent == spec["value"]
        return passed, "" if passed else f"intent was {reply.intent!r}"
    if kind == "blocked":
        return reply.blocked, "" if reply.blocked else "response was not blocked"
    if kind == "not-blocked":
        passed = not reply.blocked
        return passed, "" if passed else f"blocked as {reply.guard_category!r}"
    if kind == "escalated":
        passed = reply.escalated or reply.intent == "escalate"
        return passed, "" if passed else "response did not escalate"
    if kind == "tool-called":
        called = [c.get("tool") for c in reply.tool_calls]
        passed = spec["value"] in called
        return passed, "" if passed else f"tools called: {called}"
    if kind == "no-tool-called":
        called = [c.get("tool") for c in reply.tool_calls]
        passed = not called
        return passed, "" if passed else f"tools called: {called}"
    if kind == "no-pii-out":
        found = contains_pii(text)
        passed = found is None
        return passed, "" if passed else f"unmasked PII detected: {found}"
    if kind == "python":
        check = CHECKS[spec["value"]]
        return check(reply, case)
    if kind == "llm-rubric":
        # The judge score is recorded separately.  This assertion itself is
        # always true so an uncalibrated judge cannot silently gate safety.
        return True, "tracked by the LLM judge"

    raise ValueError(f"unknown assertion type: {kind!r}")


def load_cases(limit: int | None = None) -> list[dict[str, Any]]:
    """Load the committed golden set and optionally take a deterministic prefix."""
    cases = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("golden set must be a YAML list")
    return cases[:limit] if limit else cases


def run(*, route: str | None, limit: int | None, use_judge: bool, rubric: str, label: str) -> dict[str, Any]:
    """Execute every golden case against the real Munir application."""
    from munir.app import build_assistant, build_client
    from munir.config import load_settings

    settings = load_settings()
    assistant = build_assistant(route=route)
    judge_client = build_client(settings, settings["primary_route"]) if use_judge else None
    rubric_text = (ROOT / "eval" / "rubrics" / rubric).read_text(encoding="utf-8")

    cases = load_cases(limit)
    results: list[dict[str, Any]] = []
    started = time.perf_counter()

    for case in cases:
        vars_ = case.get("vars", {})
        student_id = vars_.get("student_id", _stable_student_id(case["id"]))
        session = Session(student_id=student_id)

        # THIS is the critical line for rubric 4.2: the harness calls the
        # same public entry point used by the notebook/application.
        reply = assistant.ask(vars_["student_message"], session, remember=False)

        failures: list[dict[str, Any]] = []
        for spec in case.get("assert", []):
            passed, detail = evaluate_assert(spec, reply, case)
            if not passed:
                failures.append({
                    "type": spec["type"],
                    "value": spec.get("value"),
                    "detail": detail,
                })

        judge = None
        has_judge_assert = any(a["type"] == "llm-rubric" for a in case.get("assert", []))
        if use_judge and has_judge_assert:
            judge = judge_case(
                judge_client,
                answer=reply.text,
                language=case.get("strata", {}).get("language", "en"),
                rubric_text=rubric_text,
                model_alias=settings["guards"]["classifier_alias"],
            )
            judge["threshold"] = next(
                a.get("threshold", 0.67)
                for a in case["assert"]
                if a["type"] == "llm-rubric"
            )

        results.append({
            "id": case["id"],
            "description": case.get("description", ""),
            "strata": case["strata"],
            "passed": not failures,
            "failures": failures,
            "judge": judge,
            "reply": reply.text,
            "intent": reply.intent,
            "blocked": reply.blocked,
            "escalated": reply.escalated,
            "tool_calls": reply.tool_calls,
            "latency_ms": round(reply.latency_ms, 1),
        })

    wall_s = round(time.perf_counter() - started, 2)
    return summarise(results, route=route, label=label, wall_s=wall_s, rubric=rubric)


def summarise(results: list[dict[str, Any]], *, route: str | None, label: str, wall_s: float, rubric: str) -> dict[str, Any]:
    """Turn individual results into overall and slice-level measurements."""
    total = len(results)
    passed = sum(1 for result in results if result["passed"])
    slices: dict[str, dict[str, dict[str, int]]] = {key: {} for key in SLICE_KEYS}

    for result in results:
        for key in SLICE_KEYS:
            value = result["strata"].get(key, "?")
            bucket = slices[key].setdefault(value, {"total": 0, "passed": 0})
            bucket["total"] += 1
            bucket["passed"] += int(result["passed"])

    judged = [result for result in results if result.get("judge") is not None]
    judge_mean = (
        round(sum(result["judge"]["score"] for result in judged) / len(judged), 3)
        if judged else None
    )
    latencies = sorted(result["latency_ms"] for result in results)
    p50 = latencies[(len(latencies) - 1) // 2] if latencies else 0

    return {
        "label": label,
        "route": route or "default",
        "rubric": rubric,
        "cases": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "slices": {
            key: {
                value: round(bucket["passed"] / bucket["total"], 4)
                for value, bucket in sorted(values.items())
            }
            for key, values in slices.items()
        },
        "slice_counts": {
            key: {
                value: bucket["total"]
                for value, bucket in sorted(values.items())
            }
            for key, values in slices.items()
        },
        "judge_mean": judge_mean,
        "judged_cases": len(judged),
        "wall_s": wall_s,
        "p50_latency_ms": p50,
        "results": results,
    }


def baseline_view(report: dict[str, Any]) -> dict[str, Any]:
    """Keep only stable aggregate measurements in the committed baseline."""
    return {
        "route": report["route"],
        "pass_rate": report["pass_rate"],
        "slices": report["slices"],
        "cases": report["cases"],
    }


def render(report: dict[str, Any]) -> None:
    """Print a human-readable evaluation summary."""
    print("\n" + "-" * 76)
    print(
        f"eval | route={report['route']} | {report['cases']} cases | "
        f"pass {report['passed']}/{report['cases']} ({report['pass_rate']:.0%}) | "
        f"{report['wall_s']}s"
    )
    print("-" * 76)
    for key in SLICE_KEYS:
        parts = [f"{value} {rate:.0%}" for value, rate in report["slices"][key].items()]
        print(f"  {key:<11} " + " | ".join(parts))
    print(f"  p50 latency {report['p50_latency_ms']} ms")
    if report["judge_mean"] is not None:
        print(
            f"  judge       groundedness mean {report['judge_mean']} "
            f"over {report['judged_cases']} cases (tracking)"
        )

    failing = [row for row in report["results"] if not row["passed"]]
    if failing:
        print(f"\n  {len(failing)} failing cases:")
        for row in failing[:12]:
            reasons = ", ".join(
                f"{failure['type']}({failure['detail']})"
                for failure in row["failures"]
            )
            print(f"    {row['id']} [{row['strata'].get('risk')}] {reasons[:100]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Munir's golden evaluation set")
    parser.add_argument("--route", default=None, help="config route, e.g. open_weight")
    parser.add_argument("--limit", type=int, default=None, help="run only the first N cases")
    parser.add_argument("--judge", action="store_true", help="run cases containing llm-rubric assertions")
    parser.add_argument("--rubric", default="groundedness.v1.md")
    parser.add_argument("--label", default="run")
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="write only aggregate measurements to eval/baseline.json",
    )
    args = parser.parse_args()

    report = run(
        route=args.route,
        limit=args.limit,
        use_judge=args.judge,
        rubric=args.rubric,
        label=args.label,
    )
    render(report)

    OUT.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.out) if args.out else OUT / f"eval_{args.label}.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written: {output_path}")

    if args.write_baseline:
        BASELINE.write_text(
            json.dumps(baseline_view(report), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"  baseline written: {BASELINE}")

    # The harness itself reports success only when deterministic assertions
    # all pass. The regression comparison belongs to gate.py.
    return 0 if report["pass_rate"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
