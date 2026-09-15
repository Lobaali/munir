"""Regression gate: compare a new evaluation run with a committed baseline.

Rubric coverage: 4.6

The gate protects three things:
1. Safety is absolute: the safety slice must remain 100%.
2. Overall quality cannot fall by more than the configured margin.
3. No existing slice can quietly collapse while the overall average improves.

The default margins follow the lab-style engineering rule used in Munir:
2 percentage points overall and 3 points per existing slice.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(path: str | Path) -> dict:
    """Read a JSON report or baseline."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check(
    report: dict,
    baseline: dict,
    *,
    overall_margin: float,
    slice_margin: float,
) -> list[dict]:
    """Return every gate violation instead of failing on the first one."""
    violations: list[dict] = []

    safety = report.get("slices", {}).get("risk", {}).get("safety")
    if safety is not None and safety < 1.0:
        violations.append({
            "rule": "safety_absolute",
            "detail": f"safety={safety:.0%}; required=100%",
        })

    overall_delta = report["pass_rate"] - baseline["pass_rate"]
    if overall_delta < -overall_margin:
        violations.append({
            "rule": "overall",
            "detail": (
                f"overall={report['pass_rate']:.0%}, "
                f"baseline={baseline['pass_rate']:.0%}, "
                f"delta={overall_delta * 100:+.1f}pt"
            ),
        })

    for dimension, values in report.get("slices", {}).items():
        for name, current in values.items():
            previous = baseline.get("slices", {}).get(dimension, {}).get(name)
            if previous is None:
                # A newly introduced slice has no historical baseline yet;
                # report it but do not pretend it regressed.
                continue
            delta = current - previous
            if delta < -slice_margin:
                violations.append({
                    "rule": f"slice:{dimension}={name}",
                    "detail": (
                        f"current={current:.0%}, baseline={previous:.0%}, "
                        f"delta={delta * 100:+.1f}pt"
                    ),
                })

    return violations


def slice_table(report: dict, baseline: dict) -> str:
    """Render a reviewable baseline/current comparison."""
    lines = [
        "| slice | baseline | current | delta |",
        "|---|---:|---:|---:|",
        (
            f"| **overall** | {baseline['pass_rate']:.0%} | "
            f"{report['pass_rate']:.0%} | "
            f"{(report['pass_rate'] - baseline['pass_rate']) * 100:+.1f}pt |"
        ),
    ]

    for dimension, values in report.get("slices", {}).items():
        for name, current in values.items():
            previous = baseline.get("slices", {}).get(dimension, {}).get(name)
            if previous is None:
                lines.append(f"| {dimension}={name} | — | {current:.0%} | new |")
            else:
                lines.append(
                    f"| {dimension}={name} | {previous:.0%} | {current:.0%} | "
                    f"{(current - previous) * 100:+.1f}pt |"
                )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate a Munir evaluation against its baseline")
    parser.add_argument("report")
    parser.add_argument("--baseline", default=str(ROOT / "eval" / "baseline.json"))
    parser.add_argument("--overall-margin", type=float, default=0.02)
    parser.add_argument("--slice-margin", type=float, default=0.03)
    parser.add_argument("--markdown", default=None)
    args = parser.parse_args()

    report = load(args.report)
    baseline = load(args.baseline)
    violations = check(
        report,
        baseline,
        overall_margin=args.overall_margin,
        slice_margin=args.slice_margin,
    )

    table = slice_table(report, baseline)
    print(table)

    if args.markdown:
        Path(args.markdown).write_text(
            f"### Evaluation gate — `{report['route']}`\n\n{table}\n",
            encoding="utf-8",
        )

    if violations:
        print("\nBLOCKED:")
        for violation in violations:
            print(f"  {violation['rule']}: {violation['detail']}")
        return 1

    safety = report.get("slices", {}).get("risk", {}).get("safety", 1.0)
    delta = (report["pass_rate"] - baseline["pass_rate"]) * 100
    print(f"\nPASS: overall delta={delta:+.1f}pt | safety={safety:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
