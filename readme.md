# Munir (منير) — Campus Services Assistant

**SDAIA Academy Capstone — SDA-AIE-213: LLM Application Engineering**

## Programme
Completed under SDAIA Academy — SDA-AIE-213: Large Language Model
Application Engineering, 13-16 Sep, 2026.
Programme repository: https://github.com/SDAIAAcademy

## Author
Loba Ali AlYahya

## What this is
A bilingual (Arabic/English) campus-services assistant admissions,
enrollment, transcripts, and advisor appointments. Built as Track B of the
capstone, chosen for three specific engineering problems:

1. **Exact-quote grounding**: tuition, deadlines, and policy facts must be
   quoted from a trusted source, never paraphrased from a model's memory.
2. **A side-effecting action**: booking an advisor appointment, gated by
   real authorization, with idempotency.
3. **Student privacy**: a student may only ever act on or view their own
   record; authorization lives in code, checked against the authenticated
   session, never in a prompt or trusted from user-claimed arguments.

## How to run

**No local install, no API key required by default.**

1. Open `notebook/munir_capstone.ipynb` in Google Colab.
2. `Runtime → Run all`.
3. The first cell clones this repo and installs dependencies. The
   assistant runs against a rule-based mock backend out of the box (see
   "About the mock backend" below).

To point at a real OpenAI-compatible provider instead, set
`MUNIR_PRIMARY_API_KEY` (as a Colab secret or environment variable) and
change `backend: mock` to `backend: real` for the relevant route in
`configs/munir.yaml`.

## About the mock backend

By default, every "model" call in this project is answered by
`MockBrainClient` (`src/munir/llm/fake.py`) a deterministic, rule-based
simulator, not a real LLM. This is disclosed openly, not hidden: it lets
the whole project run with zero cost, zero API key, and fully reproducible
results. What IS real:

- Token counting: a genuine Hugging Face tokenizer (`tokenizers`, no key
  needed), not an approximation (falls back with a loud warning if offline).
- Every guard, authorization check, cost calculation, cache lookup, and
  evaluation assertion: real code, real logic, real results.
- The model boundary itself (`LLMClient`): swapping the mock for a real
  provider is a one-line config change, proven by
  `tests/test_architecture.py` and the live fault drill in
  `EVALUATION_REPORT.md`.

## Architecture
- `LLMClient` (`src/munir/llm/interfaces.py`) is the single model boundary
   no provider SDK is imported anywhere outside
  `src/munir/llm/openai_compat.py`, enforced by
  `tests/test_architecture.py`.
- Two backends (commercial-style and open-weight-style), switchable by
  config alone.
- Router-first design: every message is classified `faq` / `service` /
  `escalate` before any generation happens.
- Authorization lives in `Session.authorize()`, checked against the
  authenticated session never against user-claimed arguments. Extends to
  read-only tools that touch per-student data.
- Full rationale in `DECISIONS.md`.

## Repository layout

| Path | Contents |
|---|---|
| `notebook/` | the capstone entry point, the only file you need to run |
| `src/munir/` | the application: llm boundary, domain, pipeline, tools, guards, caching, observability, prompts |
| `configs/munir.yaml` | model aliases, prices, guard/cache/pipeline settings — the ONE file that decides what points where |
| `data/facts/` | the trusted source of truth for grounded FAQ answers |
| `data/corpora/` | bilingual attack + legitimate test cases for the guardrails |
| `data/golden/` | the evaluation golden set, human labels for judge calibration |
| `eval/` | the harness, the regression gate, judge calibration |
| `scripts/` | guard evaluation, self-host break-even |
| `tests/` | the architecture boundary test |
| `DECISIONS.md` | architecture decision records |
| `EVALUATION_REPORT.md` | real, reproducible results from actual runs |

## Cost sources

Prices in `configs/munir.yaml` and defaults in `scripts/breakeven.py` are
real, sourced figures, not invented placeholders. All checked 2026-09-14:

| Figure | Value | Source |
|---|---|---|
| `campus-flagship` (models GPT-5.6 Sol) | $4.00 / $20.00 per MTok in/out | [BenchLM](https://benchlm.ai/openai/api-pricing), [Credit for Startups](https://creditforstartups.com/pricing/openai-api-pricing) |
| `campus-cheap` (models GPT-5.6 Luna) | $0.20 / $1.20 per MTok in/out | same sources |
| Self-hosted GPU rate (RunPod L4, on-demand) | $0.39/hr | [RunPod](https://www.runpod.io/product/cloud-gpus) |
| USD → SAR conversion | 3.75 | fixed peg, not separately sourced per-date |






