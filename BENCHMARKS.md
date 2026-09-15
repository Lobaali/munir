# BENCHMARKS.md

## Module 6 — cost and latency

The course rule is **meter first, optimise second**. The numbers in this
section must come from a real run of this repository; do not replace them with
estimates.

Run the complete replay with:

```bash
python scripts/replay.py --limit 120 --write
```

The script measures four steps in order:

1. `before` — timestamp-in-prefix prompt, no response cache, no cascade.
2. `prompt` — stable-prefix prompt, no response cache, no cascade.
3. `cache` — stable-prefix prompt + exact response cache.
4. `cascade` — stable-prefix prompt + response cache + cheap-first cascade.

The replay also runs the Section 4 golden-set harness for every step. A cost
saving is **not accepted** if the evaluation verdict regresses, especially the
safety slice.

## Before/after table

This table is intentionally a template until the replay is run on the current
machine. It is evidence only after the command above writes the measurements.

| Step | Cost / replay | Prompt-cache input share | p50 latency | Model calls | Eval pass rate | Safety rate | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| before | run | run | run | run | run | run | pending |
| prompt | run | run | run | run | run | run | pending |
| cache | run | run | run | run | run | run | pending |
| cascade | run | run | run | run | run | run | pending |

## What each optimisation proves

### Stable-prefix prompt

`answer_faq.v1` puts the current timestamp at the top of the system prompt.
That changes the prefix every request. `answer_faq.v2` moves the date into the
volatile user-turn portion, allowing an identical system prefix to be reused.

### Provider prompt caching

The model boundary exposes `cached_input_tokens`. The local `MockBrainClient`
observes repeated byte-identical system prefixes and reports the reused token
count through the same `Usage` field a real provider would populate.

This is a measurement mechanism, not a claim that the local mock has a real
GPU/provider cache.

### Exact response caching

`ResponseCache.exact_key()` includes the model alias, prompt version, rendered
trusted context, user question, language, conversation history, and sampling
parameters. Leaving out any of those could serve an answer produced under
different rules.

Personalised service conversations are deliberately not served from this FAQ
cache.

### Cheap-first cascade

The FAQ handler can call the cheap model first. If the deterministic
`unsupported_amounts()` check finds a fee the trusted directory does not
contain, Munir escalates that turn to the flagship model. The meter records
both calls, so an escalation is not falsely reported as a saving.

## Important limitation

The repository currently uses `MockBrainClient` for offline reproducibility.
Its timing and generated answers are simulated. The engineering evidence that
matters here is real: usage is converted into cost records, cached tokens are
observed, the cache key is tested, and optimisation steps are evaluation-gated.
For final submission evidence, rerun the same replay against the real provider
or measured open-weight deployment when those are available.
