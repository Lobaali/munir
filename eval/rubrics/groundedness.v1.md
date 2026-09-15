# Groundedness rubric — v1

## Dimension being judged

Judge **groundedness only**: whether factual claims in the answer are
supported by the supplied trusted campus directory.

Do not score tone, helpfulness, grammar, or general safety here. Those are
separate concerns and should not be hidden inside one broad judge score.

## Context

The `<context>` section is the only trusted source of campus facts.

## Scoring

### 1.0 — Fully grounded
Every factual fee, deadline, document requirement, or service fact stated in
the answer is supported by the context and does not contradict it.

### 0.5 — Safe but incomplete
The answer is vague, partial, or appropriately says it does not know. It does
not introduce a contradictory or invented campus fact.

### 0.0 — Ungrounded
The answer invents a fee, date, requirement, or other campus fact, or it
contradicts the supplied context.

## Required output

Return exactly one JSON object matching the requested schema:

```json
{"score": 1.0, "evidence": "One short sentence explaining the score."}
```

The score must be exactly `0.0`, `0.5`, or `1.0`.
