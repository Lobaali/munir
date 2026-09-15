# Munir Evaluation Harness — Section 4

This directory implements the evaluation requirements from the course's
"harness, judge, and gate" lab.

## Flow

```text
regression_set.yaml
        |
        v
   eval/harness.py
        |
        +--> deterministic assertions  ----> PASS/FAIL
        |
        +--> optional LLM judge ---------> tracking metric
        |
        v
   eval/out/eval_*.json
        |
        v
      gate.py  <----  baseline.json
        |
      PASS / BLOCK
```

## Files

- `golden/regression_set.yaml` — 120 Munir-specific golden cases with four
  strata: language, intent, difficulty, and risk.
- `asserts/checks.py` — deterministic checks used for safety and grounding.
- `judge.py` — one-dimensional groundedness judge with a strict Pydantic
  response contract.
- `rubrics/groundedness.v1.md` — versioned written judge rubric.
- `golden/human_labels.jsonl` — human calibration sample.
- `calibrate_judge.py` — exact agreement + Cohen's kappa.
- `gate.py` — baseline/slice regression gate with an absolute safety rule.
- `baseline.json` — committed aggregate baseline. Regenerate it after the
  first real run with `--write-baseline` and commit the measured result.

## Commands

Run deterministic evaluation:

```bash
python eval/harness.py --label current
```

Run the judge on the representative `llm-rubric` cases:

```bash
python eval/harness.py --judge --label current_with_judge
```

Calibrate the judge:

```bash
python eval/calibrate_judge.py --rubric groundedness.v1.md
```

Create/update the committed baseline from a real run:

```bash
python eval/harness.py --label baseline --write-baseline
```

Gate a later run against the baseline:

```bash
python eval/gate.py eval/out/eval_current.json --baseline eval/baseline.json
```

The gate always blocks if `risk=safety` drops below 100%, even if the overall
score improves.
