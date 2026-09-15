# Munir (منير) — Campus Services Assistant

**SDAIA Academy Capstone — SDA-AIE-213: LLM Application Engineering**

## Programme

Completed under SDAIA Academy — SDA-AIE-213: Large Language Model Application Engineering, 13–16 Sep, 2026.

Programme repository: https://github.com/SDAIAAcademy

## Author

Loba Ali AlYahya

## What this is

A bilingual (Arabic/English) campus-services assistant for admissions, enrollment, transcripts, and advisor appointments. Built as Track B of the capstone, chosen for three specific engineering problems:

1. **Exact-quote grounding**: tuition, deadlines, and policy facts must be quoted from a trusted source, never paraphrased from a model's memory.

2. **A side-effecting action**: booking an advisor appointment, gated by real authorization, with idempotency.

3. **Student privacy**: a student may only ever act on or view their own record; authorization lives in code, checked against the authenticated session, never in a prompt or trusted from user-claimed arguments.

## How to run

**No local API key is required by default.**

The recommended entry point is:

```text
notebook/munir_run_all.ipynb
```

Open the notebook in Jupyter or Google Colab and select:

```text
Kernel → Restart Kernel and Run All
```

The notebook runs the project end-to-end, including:

* dependency installation
* automated tests
* the 120-case golden-set evaluation
* LLM-judge calibration
* regression gate
* cost and latency replay
* commercial-vs-open-weight comparison
* self-host break-even calculation
* generated evaluation artifacts

The project runs against the deterministic mock backend by default, so no API key is required.

### Running from the command line

The individual commands can also be run directly from the repository root:

```bash
PYTHONPATH=.:src python3 -m pytest -q
```

```bash
PYTHONPATH=.:src python3 scripts/replay.py --limit 120 --write
```

```bash
PYTHONPATH=.:src python3 scripts/compare_models.py --limit 120
```

```bash
PYTHONPATH=.:src python3 scripts/breakeven.py \
  --gpu-usd-per-hour 3.33 \
  --tokens-per-sec 950 \
  --utilization 0.50 \
  --commercial-price-per-mtok 15 \
  --avg-tokens-per-request 100
```

## About the mock backend

By default, model calls are answered by `MockBrainClient` (`src/munir/llm/fake.py`), a deterministic, rule-based simulator rather than a real LLM.

This is disclosed openly. It allows the complete project to run with zero cost, zero API key, and reproducible results.

What **is** real:

* Token counting uses the project's tokenizer implementation.
* Guard checks are real code.
* Authorization checks are real code.
* Cost calculations are real code.
* Cache lookup and cache-key logic are real code.
* Evaluation assertions are real code.
* The model boundary (`LLMClient`) is real and provider-independent.
* Commercial-style and open-weight-style routes are exercised through the same application and golden set.

Mock benchmark latency and throughput must not be interpreted as production-provider measurements.

## Architecture

* `LLMClient` (`src/munir/llm/interfaces.py`) is the single model boundary.
* Provider-specific SDK access is isolated from business logic.
* Model routes are selected through configuration.
* Two backends are provided: commercial-style and open-weight-style.
* Router-first design: every message is classified as `faq`, `service`, or `escalate` before generation.
* Authorization lives in `Session.authorize()`, checked against the authenticated session rather than user-claimed arguments.
* Side-effecting tools use authorization and idempotency protections.
* Full architectural decisions are documented in `DECISIONS.md`.

## Repository layout

| Path                   | Contents                                                                                    |
| ---------------------- | ------------------------------------------------------------------------------------------- |
| `notebook/`            | Main capstone entry point and Run All notebook                                              |
| `src/munir/`           | Application: LLM boundary, domain, pipeline, tools, guards, caching, observability, prompts |
| `configs/munir.yaml`   | Model aliases, prices, guard/cache/pipeline settings                                        |
| `data/facts/`          | Trusted source of truth for grounded FAQ answers                                            |
| `data/corpora/`        | Bilingual attack and legitimate test cases                                                  |
| `data/golden/`         | Evaluation golden set and human labels                                                      |
| `eval/`                | Evaluation harness, judge calibration, regression gate                                      |
| `scripts/`             | Replay, model comparison, and self-host break-even scripts                                  |
| `tests/`               | Automated tests                                                                             |
| `DECISIONS.md`         | Architecture decision records                                                               |
| `EVALUATION_REPORT.md` | Evaluation results, calibration, regression, cost/cache evidence, and limitations           |
| `BENCHMARKS.md`        | Benchmark and commercial/open-weight comparison evidence                                    |

## Evaluation

The evaluation uses a versioned 120-case golden set covering:

* Arabic and English
* FAQ, service, and escalation intents
* routine, edge, and adversarial difficulty
* normal and safety-sensitive cases

The project also includes human labels for judge calibration and a regression gate against a committed baseline.

The complete evaluation evidence is documented in:

```text
EVALUATION_REPORT.md
```

## Commercial vs open-weight

Both model routes were exercised against the same 120-case golden set.

The latest comparison produced:

| Route                       |  Passed | Pass Rate |       Cost |
| --------------------------- | ------: | --------: | ---------: |
| Commercial (`primary`)      | 119/120 |       99% | 1.6328 SAR |
| Open-weight (`open_weight`) | 117/120 |       98% | 0.1690 SAR |

Both routes achieved 100% on the safety slice.

The largest difference was in the edge slice:

* Commercial: 100%
* Open-weight: 93%

The open-weight route was approximately 89.6% cheaper in this benchmark.

## Self-host break-even

The break-even calculation currently uses a documented benchmark assumption of **950 output tokens/second** rather than a measured throughput value from the mock environment.

Using:

* GPU rate: $3.33/hour
* Utilization: 50%
* Throughput assumption: 950 output tok/s
* Commercial price: 15 SAR / 1M tokens
* Average request: 100 tokens

the calculation produces:

* Derived self-host cost: 7.3026 SAR / 1M tokens
* Monthly GPU rental: 8,991 SAR
* Break-even volume: approximately 5.994 million requests/month

The 950 tok/s value must therefore be treated as an assumption, not production throughput evidence.

## Cost sources

Prices used by the project are documented in `configs/munir.yaml` and `BENCHMARKS.md`.

The project distinguishes measured/observed benchmark results from documented assumptions. In particular, the 950 tok/s self-host throughput value is explicitly documented as an assumption.

## Limitations

The primary limitation of this submission is that the default model backend is a deterministic mock rather than a live production LLM.

Therefore:

* Mock latency is not production latency.
* Mock throughput is not production throughput.
* The 950 tok/s self-host value is an assumption rather than a current deployment measurement.
* Commercial/open-weight quality comparisons demonstrate the application's evaluation pipeline and route comparison using the configured backends, but should not be interpreted as a production-model benchmark.

These limitations are intentionally disclosed rather than hidden.
