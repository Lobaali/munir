# ADR 001 — Router-first, with one bounded tool loop

## Context

Munir's traffic splits into three shapes: factual questions answerable from a fixed set of facts (tuition, deadlines, required documents), transactional requests that need a tool and an authorization check (booking an advisor appointment, checking enrollment status, requesting a transcript), and requests that need a human (grade disputes, appeals).

There's no replay log to measure the real split from yet — this is a capstone, not a live service — but the shape of the three categories themselves is what's being decided on here, not their exact proportions.

Three patterns were on the table: a single agentic loop for everything, a fixed workflow for everything, or a router dispatching to specialised handlers.

## Decision

**Router first.** A classifier assigns one of three intents and dispatches:

* `faq` → a single call against the trusted facts file, cacheable, cheap model first, escalating to the flagship model only if the cheap answer isn't grounded;
* `service` → a fixed workflow whose middle is a **bounded** tool loop (6 iterations, an allowed-tool list, an authorisation gate on any tool that touches student data — not only side-effecting ones);
* `escalate` → a human. Not a model call at all.

The tool loop is the only agentic component in the project, and it's bounded on every axis that can run away: iterations, the tool list, and a trace of every call.

### Model boundary and model choices

The application does not call a provider SDK directly from business logic. All model calls go through the typed `LLMClient` boundary defined in `munir/llm/interfaces.py`.

Model identifiers are configuration concerns rather than application-code constants. Munir uses model aliases in `configs/munir.yaml`, allowing the selected provider/model to be changed without changing the routing or business logic.

The model strategy follows the traffic shape:

* **Router/guard tasks** use the lower-cost configured model because these tasks require classification or safety decisions rather than long-form generation.
* **FAQ generation** starts with the lower-cost configured model and can escalate to the configured flagship model when the groundedness check fails.
* **Structured extraction/tool decisions** use the configured model assigned to that task.
* **Escalation requests** do not invoke an LLM because they are intentionally routed to a human.

Provider-specific SDK code is isolated behind the LLM adapter. This keeps the application independent of a particular provider and allows the same application boundary to be exercised with the configured provider or test/fake client.

Retries and fallback behaviour are also implemented at the model boundary rather than inside individual application handlers.

## Consequences

**Good.** Each path is testable in isolation. Cost follows traffic shape rather than worst case: the FAQ path costs a fraction of the flagship path, and FAQ-shaped questions are expected to dominate real usage. A misroute is **measurable** — it's a stratum in the golden set — where an agent's wrong turn is a transcript somebody has to read by hand.

**Good.** Model selection is replaceable without rewriting application logic. Model IDs can change through configuration, while the rest of the application continues to depend on the typed model interface.

**Bad.** Two more moving parts than a single call, and a router that's wrong sends a student down the wrong path. Accepted because the misroute rate is measured and gated (see `eval/harness.py`'s per-slice reporting), and because the failure is legible: a wrong route produces a wrong-shaped answer, not a quietly expensive loop.

**Bad.** Using multiple model tiers adds configuration and evaluation overhead. The cheaper model may occasionally require escalation, increasing latency and cost for those requests. This is accepted because the escalation decision is bounded and can be evaluated on the golden set.

## Rejected: agent-first

The decomposition here is known at design time — three categories, three handlers. An agentic loop would buy flexibility this project doesn't need and pay for it in unbounded cost, undebuggable traces, and no clean way to bound a booking action's autonomy.

The rule: start at the simplest pattern the use case allows; escalate on evidence, not on vocabulary.

## Rejected: workflow-only

A fixed chain for FAQ traffic would run tool schemas and workflow-prompt overhead for every "how much does a transcript cost?" — paying the transactional price for conversational traffic that never needs a tool at all.

## Rejected: hardcoded provider/model calls

Putting provider SDK calls and model IDs directly into application handlers would make provider changes expensive and couple business logic to infrastructure. This was rejected in favour of the typed `LLMClient` boundary and configuration-based model aliases.

## Implementation mapping

* **Architecture/routing:** `munir/app.py`, `munir/pipeline/`
* **Typed model boundary:** `munir/llm/interfaces.py`
* **Provider adapter:** `munir/llm/openai_compat.py`
* **Resilience:** `munir/llm/resilient.py`
* **Model configuration:** `configs/munir.yaml`
* **Evaluation of routing and model behaviour:** `eval/harness.py`
