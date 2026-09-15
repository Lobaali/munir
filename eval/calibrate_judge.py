"""Calibrate the LLM judge against human labels.

Rubric coverage: 4.5

We report two numbers:
- exact agreement: how often the judge selected the same score as a human;
- Cohen's kappa: agreement corrected for the agreement expected by chance.

If kappa is below the configured bar, the command exits non-zero.  The
intended response is to improve the versioned rubric, not to lower the bar.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "eval") not in sys.path:
    sys.path.insert(0, str(ROOT / "eval"))

from eval.judge import judge_case  # noqa: E402

LABELS = ROOT / "eval" / "golden" / "human_labels.jsonl"
OUT = ROOT / "eval" / "out"


def cohen_kappa(human: list[float], judged: list[float]) -> float:
    """Calculate Cohen's kappa without adding a heavy ML dependency."""
    if len(human) != len(judged) or not human:
        raise ValueError("kappa requires two equally sized non-empty label lists")

    categories = sorted(set(human) | set(judged))
    n = len(human)
    observed = sum(h == j for h, j in zip(human, judged, strict=True)) / n
    expected = sum(
        (human.count(category) / n) * (judged.count(category) / n)
        for category in categories
    )

    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate Munir's LLM judge")
    parser.add_argument("--rubric", default="groundedness.v1.md")
    parser.add_argument("--route", default=None)
    parser.add_argument("--bar", type=float, default=0.60)
    args = parser.parse_args()

    from munir.app import build_client
    from munir.config import load_settings

    settings = load_settings()
    route = args.route or settings["primary_route"]
    client = build_client(settings, route)
    rubric_text = (ROOT / "eval" / "rubrics" / args.rubric).read_text(encoding="utf-8")

    rows = [
        json.loads(line)
        for line in LABELS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("human label file is empty")

    human_scores: list[float] = []
    judge_scores: list[float] = []
    disagreements: list[dict] = []

    for row in rows:
        verdict = judge_case(
            client,
            answer=row["answer"],
            language=row["language"],
            rubric_text=rubric_text,
            model_alias=settings["guards"]["classifier_alias"],
        )
        human_score = float(row["human_score"])
        judge_score = float(verdict["score"])
        human_scores.append(human_score)
        judge_scores.append(judge_score)

        if human_score != judge_score:
            disagreements.append({
                "case_id": row["case_id"],
                "human": human_score,
                "judge": judge_score,
                "evidence": verdict["evidence"],
            })

    agreement = sum(h == j for h, j in zip(human_scores, judge_scores, strict=True)) / len(rows)
    kappa = cohen_kappa(human_scores, judge_scores)

    result = {
        "rubric": args.rubric,
        "route": route,
        "cases": len(rows),
        "agreement": round(agreement, 4),
        "cohen_kappa": round(kappa, 4),
        "bar": args.bar,
        "passes_bar": kappa >= args.bar,
        "disagreements": disagreements,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / f"calibration_{Path(args.rubric).stem}.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\nrubric: {args.rubric}")
    print(f"  exact agreement: {agreement:.0%}")
    print(f"  Cohen's kappa:   {kappa:.2f}")
    print(f"  cases:           {len(rows)}")
    print(f"  calibration:     {'PASS' if kappa >= args.bar else 'FAIL'} (bar={args.bar:.2f})")
    print(f"  written:         {output}")

    if disagreements:
        print("\n  Disagreements to inspect:")
        for item in disagreements[:10]:
            print(
                f"    {item['case_id']}: human={item['human']} "
                f"judge={item['judge']} -- {item['evidence']}"
            )

    return 0 if kappa >= args.bar else 1


if __name__ == "__main__":
    raise SystemExit(main())
