# Munir — Architecture Decisions

This file records the main architectural decisions made for the Munir capstone.

Detailed rationale for the architecture and model-boundary decision is documented in:

* `docs/adr/001-architecture-pattern.md`

---

## Decision 1 — Router-first architecture

**Status:** Accepted

Munir uses a **router-first architecture** rather than one general-purpose agent.

The router classifies requests into three intents:

* `faq` — factual questions that can be answered from trusted campus information.
* `service` — requests that require a workflow and potentially a tool.
* `escalate` — requests that should be handled by a human.

### Why

The three request categories are known at design time. A router makes each path easier to test, evaluate, secure, and control.

A fully agentic architecture was rejected because it would introduce unnecessary autonomy, cost, and debugging complexity.

A workflow-only architecture was rejected because simple FAQ questions should not pay the cost of transactional workflows and tool handling.

---

## Decision 2 — One bounded tool loop

**Status:** Accepted

The `service` path may use a tool loop, but the loop is explicitly bounded.

The loop is limited by:

* a maximum of **6 iterations**;
* an explicit allowed-tool list;
* authorization checks before tools that access student data;
* recording each tool call in the execution trace.

### Why

Tool use is necessary for transactional requests, but unrestricted agentic execution would make cost, latency, and side effects harder to control.

The bounded loop provides the flexibility of tool calling while keeping execution predictable and testable.

---

## Decision 3 — Typed LLM boundary

**Status:** Accepted

Application and business logic depend on the typed `LLMClient` interface rather than directly importing a provider SDK.

The main model boundary is defined in:

`munir/llm/interfaces.py`

Provider-specific implementation is isolated in:

`munir/llm/openai_compat.py`

### Why

This prevents provider-specific implementation details from spreading through the application.

It also allows Munir to use different model implementations, including the fake client used for deterministic tests, without changing business logic.

---

## Decision 4 — Configuration-based model selection

**Status:** Accepted

Model identifiers are configuration values rather than hardcoded throughout the application.

Model configuration is stored in:

`configs/munir.yaml`

The application refers to configured model aliases/roles instead of embedding provider model IDs directly inside business logic.

### Why

Model choices may change during development and evaluation. Keeping them in configuration allows the model strategy to change without rewriting the application.

This also makes it possible to compare different model tiers during evaluation.

---

## Decision 5 — Model tiers follow the traffic shape

**Status:** Accepted

Munir uses the configured lower-cost model for tasks that do not require the strongest generation capability, while the configured flagship model is available when higher-quality generation is required.

For example:

* routing/classification → lower-cost configured model;
* FAQ generation → lower-cost model first;
* FAQ escalation → configured flagship model when the initial answer fails the groundedness requirement;
* human escalation → no model call.

### Why

Not every request requires the same model capability.

Using a cheaper model for simpler tasks reduces cost and latency, while keeping a stronger model available for cases where the cheaper model does not produce an acceptable answer.

The actual provider model IDs remain in `configs/munir.yaml` rather than being duplicated in this document.

---

## Decision 6 — Resilience belongs at the model boundary

**Status:** Accepted

Retry, backoff, and fallback behaviour are implemented at the LLM boundary rather than separately inside each application handler.

Implementation:

`munir/llm/resilient.py`

### Why

Provider failures are infrastructure concerns. Keeping resilience at the model boundary means every model-using part of Munir receives consistent retry and fallback behaviour.

It also prevents individual handlers from implementing different retry policies.

---

## Decision 7 — Trusted facts are separated from generated answers

**Status:** Accepted

FAQ responses are generated using a trusted campus facts source rather than allowing the model to invent institutional information.

The trusted data is stored under:

`data/campus_directory.yaml`

and is supplied to the appropriate application path as trusted context.

### Why

Questions about tuition, deadlines, documents, and campus services require factual grounding.

Separating trusted facts from generated language makes it possible to evaluate whether the answer is supported by the available information.

---

## Decision 8 — Evaluation gates architectural changes

**Status:** Accepted

Changes to routing, models, prompts, caching, and other optimization strategies should be evaluated against the versioned golden set before being considered improvements.

The evaluation harness is located under:

`eval/`

### Why

A change that reduces cost or latency is not automatically an improvement if it reduces safety or answer quality.

Munir therefore treats evaluation results as a gate for optimization decisions.

---

## Decision summary

| Decision          | Choice                                                  |
| ----------------- | ------------------------------------------------------- |
| Architecture      | Router-first                                            |
| Agentic behaviour | One bounded tool loop                                   |
| LLM integration   | Typed `LLMClient` boundary                              |
| Provider SDK      | Isolated in adapter                                     |
| Model IDs         | Configuration                                           |
| Model strategy    | Lower-cost model first, flagship escalation when needed |
| Resilience        | Retry/backoff/fallback at LLM boundary                  |
| FAQ knowledge     | Trusted campus facts                                    |
| Evaluation        | Versioned golden set + regression gate                  |

For the detailed rationale, alternatives considered, and consequences, see:

`docs/adr/001-architecture-pattern.md`
