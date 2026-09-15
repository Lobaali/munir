# Munir — Evaluation Report

**SDAIA Academy Capstone — SDA-AIE-213: LLM Application Engineering**

**Author:** Loba Ali AlYahya
**Evaluation date:** September 2026

---

## 1. Executive Summary

Munir is a bilingual Arabic/English campus-services assistant designed around grounded answers, protected student data, and authorized side-effecting actions.

The project was evaluated using a versioned golden set, automated assertions, safety checks, judge calibration, regression testing, cost/cache replay, and commercial-vs-open-weight route comparison.

The default project backend is a deterministic mock model backend. This is intentional and is disclosed throughout the repository. The mock backend allows the complete evaluation pipeline to run deterministically without an API key or external model cost.

The application and evaluation infrastructure are real code. The model responses produced by the mock backend are not production LLM measurements.

---

## 2. Evaluation Setup

### Golden set

The current golden set contains **120 cases**.

| Dimension   | Cases |
| ----------- | ----: |
| Arabic      |    52 |
| English     |    68 |
| FAQ         |    73 |
| Service     |    33 |
| Escalation  |    14 |
| Routine     |    57 |
| Edge        |    29 |
| Adversarial |    34 |
| Normal      |    86 |
| Safety      |    34 |

The golden set is versioned in:

```text
eval/golden/regression_set.yaml
```

Human labels used for judge calibration are stored in:

```text
eval/golden/human_labels.jsonl
```

---

## 3. Automated Test Suite

The project includes automated tests covering the application architecture, guards, pipeline behavior, evaluation infrastructure, cost/cache behavior, and other rubric requirements.

The complete suite is run with:

```bash
PYTHONPATH=.:src python3 -m pytest -q
```

Latest full-suite result:

```text
38 passed
```

---

## 4. Evaluation Harness

The evaluation harness runs the application against the versioned golden set and applies deterministic assertions.

The evaluation covers:

* expected intent
* expected language
* groundedness
* safety behavior
* escalation behavior
* supported/unsupported service information
* required output structure

The harness is implemented in:

```text
eval/harness.py
eval/asserts/checks.py
```

The golden set is declarative and versioned, allowing the same cases to be replayed after application changes.

---

## 5. Safety Evaluation

The golden set contains **34 safety-sensitive cases**.

The latest commercial-style and open-weight-style route comparison produced:

| Route                       | Safety Pass Rate |
| --------------------------- | ---------------: |
| Commercial (`primary`)      |             100% |
| Open-weight (`open_weight`) |             100% |

Safety behavior is enforced through deterministic assertions rather than relying only on an LLM judge.

The safety layer also includes input and output guardrails for prompt injection, sensitive information, and outbound leakage.

---

## 6. Commercial vs Open-Weight Evaluation

Both routes were exercised against the same 120-case golden set using:

```bash
PYTHONPATH=.:src python3 scripts/compare_models.py --limit 120
```

### Overall results

| Route                       |  Passed | Pass Rate | Cost (SAR) | Model Calls |
| --------------------------- | ------: | --------: | ---------: | ----------: |
| Commercial (`primary`)      | 119/120 |       99% |     1.6328 |         335 |
| Open-weight (`open_weight`) | 117/120 |       98% |     0.1690 |         336 |

### Slice comparison

| Slice       | Commercial | Open-weight | Delta |
| ----------- | ---------: | ----------: | ----: |
| Arabic      |       100% |         98% |   -2% |
| English     |        99% |         97% |   -1% |
| Escalation  |       100% |        100% |    0% |
| FAQ         |       100% |         97% |   -3% |
| Service     |        97% |         97% |    0% |
| Adversarial |       100% |        100% |    0% |
| Edge        |       100% |         93% |   -7% |
| Routine     |        98% |         98% |    0% |
| Normal      |        99% |         97% |   -2% |
| Safety      |       100% |        100% |    0% |

The commercial route scored 99% overall, compared with 98% for the open-weight route.

The largest difference occurred in the edge slice, where the commercial route achieved 100% and the open-weight route achieved 93%.

The open-weight route was substantially cheaper in this benchmark:

* Commercial: 1.6328 SAR
* Open-weight: 0.1690 SAR
* Approximate reduction: 89.6%

Both routes maintained 100% safety performance.

---

## 7. Cost and Latency Optimization

The project includes a replay script that compares optimization stages while checking that evaluation quality and safety are preserved.

Command:

```bash
PYTHONPATH=.:src python3 scripts/replay.py --limit 120 --write
```

Latest recorded comparison:

| Step    | Cost (SAR) | Cached Input | p50 Latency (ms) | Model Calls | Eval | Safety |
| ------- | ---------: | -----------: | ---------------: | ----------: | ---: | -----: |
| Before  |   0.189458 |        70.8% |              0.9 |          35 | 100% |   100% |
| Prompt  |   0.189458 |        70.8% |              0.7 |          35 | 100% |   100% |
| Cache   |   0.117870 |        64.3% |              0.8 |          31 | 100% |   100% |
| Cascade |   0.117870 |        64.3% |              0.8 |          31 | 100% |   100% |

The cache stage reduced the recorded cost from 0.189458 SAR to 0.117870 SAR, a reduction of approximately **37.78%**, while preserving 100% evaluation and safety performance.

The replay results are written to:

```text
eval/out/cost_optimization_comparison.json
```

---

## 8. Prompt and Cache Discipline

The project separates stable prompt content from volatile request content.

Stable prompt content is kept in versioned prompt artifacts, while request-specific information is placed in the volatile portion of the request.

The project also uses:

* versioned prompt artifacts
* deterministic prompt injection patterns
* response caching
* answer-changing variables in cache keys
* cache-prefix tracking

The cache implementation is designed so that changing the user query, language, history, prompt version, model route, or other answer-changing variables does not incorrectly reuse an unrelated response.

---

## 9. Self-Hosted Break-Even Analysis

The self-host break-even calculation was executed with:

```bash
PYTHONPATH=.:src python3 scripts/breakeven.py \
  --gpu-usd-per-hour 3.33 \
  --tokens-per-sec 950 \
  --utilization 0.50 \
  --commercial-price-per-mtok 15 \
  --avg-tokens-per-request 100
```

Result:

| Metric                 |                     Value |
| ---------------------- | ------------------------: |
| GPU hourly rate        |                $3.33/hour |
| Utilization            |                       50% |
| Throughput assumption  |          950 output tok/s |
| Derived self-host cost |    7.3026 SAR / 1M tokens |
| Monthly GPU rental     |                 8,991 SAR |
| Commercial cost        |   15.0000 SAR / 1M tokens |
| Average request size   |                100 tokens |
| Break-even volume      | ~5,994,000 requests/month |


---

## 10. Mock Backend Disclosure

The default backend is:

```text
src/munir/llm/fake.py
```

and uses `MockBrainClient`.

The mock backend is deterministic and rule-based.

This means:

### Real and evaluated

* application pipeline
* model boundary
* provider abstraction
* routing
* structured output validation
* tool definitions
* authorization
* guardrails
* PII protection
* prompt pipeline
* evaluation harness
* deterministic assertions
* judge calibration infrastructure
* regression gate
* cost accounting
* cache behavior
* commercial/open-weight route comparison logic

### Not production-model evidence

* model quality of a real commercial provider
* real provider latency
* real provider throughput
* production self-host throughput

This distinction is intentional and is documented in the README.

---

## 11. Known Limitations

1. The default model backend is a deterministic mock rather than a live production LLM.
2. Mock latency and throughput should not be used as production performance claims.
3. The 950 tok/s value used for break-even analysis is a documented assumption.
4. A real deployment should rerun the throughput and cost measurements against the intended provider and self-hosted model server before making production capacity or economic decisions.

---

## 12. Reproducibility

The recommended entry point is:

```text
notebook/munir_run_all.ipynb
```

The notebook runs the main project checks and evaluation commands in one place.

The individual commands can also be executed directly from the repository root.

The project intentionally defaults to the deterministic mock backend so that the complete evaluation can be reproduced without an API key.

---

## 13. Final Assessment

The current implementation provides:

* a typed LLM boundary
* configurable model routes
* structured output validation
* bounded tool execution
* authorization for side-effecting operations
* versioned prompt artifacts
* bilingual guardrails
* PII protection
* a versioned golden-set evaluation harness
* deterministic safety assertions
* judge calibration infrastructure
* regression testing
* cost and cache analysis
* commercial/open-weight comparison
* self-host break-even analysis
* a single notebook entry point for reproducible execution

The principal limitation is the use of a deterministic mock backend for the default run. This limitation is explicitly disclosed, and the repository distinguishes application/evaluation evidence from real production-model measurements.
