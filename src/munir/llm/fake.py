"""``FakeClient`` -- the third implementation of the boundary, and the one that
makes the test suite fast, free and deterministic.

It is scripted, not clever: you tell it what to return, in order. Everything a
walkthrough needs to test -- a tool call, a malformed argument, an endless loop, a 429
storm, a schema violation the repair loop must fix -- is a one-line script here.

This is the payoff of the LLMClient Protocol you feel first: you can
test the whole application without a network, a key, or a bill.
"""

from __future__ import annotations

import itertools
import json
import time
from collections.abc import Callable, Iterator

from munir.llm.interfaces import (
    LLMError,
    LLMRequest,
    LLMResponse,
    StreamChunk,
    ToolCall,
    Usage,
)


class FakeClient:
    def __init__(self, model_id: str = "fake-model", route: str = "fake") -> None:
        self.model_id = model_id
        self.route = route
        self._queue: list[LLMResponse | Exception] = []
        self._default: Callable[[LLMRequest], LLMResponse] | None = None
        self._repeat: LLMResponse | Exception | None = None
        self.requests: list[LLMRequest] = []
        self._ids = itertools.count(1)

    # --- scripting -------------------------------------------------------
    def script_text(self, text: str, *, finish: str = "stop", tokens: tuple[int, int] = (100, 40)):
        self._queue.append(
            LLMResponse(
                text=text,
                finish_reason=finish,
                model_id=self.model_id,
                usage=Usage(input_tokens=tokens[0], output_tokens=tokens[1]),
                latency_ms=1.0,
                route=self.route,
            )
        )
        return self

    def script_json(self, payload: dict | str, **kwargs):
        text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        return self.script_text(text, **kwargs)

    def script_tool_call(self, name: str, arguments: dict | str, *, text: str = ""):
        args = arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
        self._queue.append(
            LLMResponse(
                text=text or None,
                tool_calls=[ToolCall(id=f"call_{next(self._ids)}", name=name, arguments=args)],
                finish_reason="tool_calls",
                model_id=self.model_id,
                usage=Usage(input_tokens=120, output_tokens=30),
                latency_ms=1.0,
                route=self.route,
            )
        )
        return self

    def script_endless_tool_calls(self, name: str, arguments: dict | None = None):
        """The stubborn model. Bounds are the only thing that stops it."""
        args = json.dumps(arguments or {"reference": "CR12345678"}, ensure_ascii=False)
        self._repeat = LLMResponse(
            tool_calls=[ToolCall(id="call_loop", name=name, arguments=args)],
            finish_reason="tool_calls",
            model_id=self.model_id,
            usage=Usage(input_tokens=120, output_tokens=30),
            latency_ms=1.0,
            route=self.route,
        )
        return self

    def script_error(self, error: Exception, times: int = 1):
        for _ in range(times):
            self._queue.append(error)
        return self

    def script_rate_limit(self, times: int = 1, retry_after: float | None = 2.0):
        return self.script_error(
            LLMError("429 rate limit", status=429, retryable=True, retry_after=retry_after),
            times=times,
        )

    def always(self, fn: Callable[[LLMRequest], LLMResponse]):
        """Fall back to a function of the request when the queue empties."""
        self._default = fn
        return self

    # --- the LLMClient protocol -------------------------------------------
    @property
    def route_name(self) -> str:
        return self.route

    @property
    def call_count(self) -> int:
        return len(self.requests)

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self._queue:
            item = self._queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        if self._repeat is not None:
            if isinstance(self._repeat, Exception):
                raise self._repeat
            return self._repeat.model_copy()
        if self._default is not None:
            return self._default(request)
        raise AssertionError(
            "FakeClient ran out of scripted responses -- script one more, or set .always()"
        )

    def stream(self, request: LLMRequest) -> Iterator[StreamChunk]:
        response = self.complete(request)
        t0 = time.perf_counter()
        for word in (response.text or "").split(" "):
            yield StreamChunk(delta=word + " ", model_id=response.model_id)
        yield StreamChunk(
            final=True,
            usage=response.usage,
            ttft_ms=0.5,
            total_ms=(time.perf_counter() - t0) * 1000,
            model_id=response.model_id,
            finish_reason=response.finish_reason,
        )


# ---------------------------------------------------------------------------
# MockBrainClient -- a diverging design choice from Murshid, documented
# honestly rather than silently: Murshid answers "fake" free-form traffic
# with a separate HTTP mock-gateway SERVICE (infra/mockgw), simulating a
# real provider's wire format over the network. This project instead uses
# a plain Python class satisfying LLMClient directly -- no server, no
# network, same "no key, no cost, deterministic" property, simpler to run
# in a notebook with nothing else to start up. See DECISIONS.md.
#
# It is NOT intelligent. It pattern-matches against the trusted directory
# text embedded in the prompt, and it CANNOT invent a fact that wasn't
# handed to it in that same prompt -- it isn't generating language, it's
# templating off what's there. That's disclosed everywhere this class is
# used, not just here.
# ---------------------------------------------------------------------------

import random
import re

from munir.domain import directory
from munir.llm.tokens import count as ntokens_for_model

def extract_tag(text: str, tag: str) -> str | None:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else None


from dataclasses import dataclass

#: Marker phrase the brain looks for in the system prompt. If present, the
#: brain admits "I don't know" on out-of-facts questions. If ABSENT, it
#: invents an answer at least 60% of the time. This is the exact mechanism
#: that lets prompt WORDING changes be meaningfully tested -- this one
#: phrase is the load-bearing part of the grounding instruction.
DONT_KNOW_MARKERS = ["say you do not know", "say you don't know", "لا تعرف"]

#: Deliberately worse behavior for the "cheap"/"open-weight" tier.
TIER_GUESS_RATE = {
    "campus-flagship": 0.0,
    "campus-cheap": 0.15,
    "campus-onprem": 0.30,
}

INVENTED_FEES = ["SAR 350", "SAR 275", "SAR 120"]


@dataclass
class MockBrainClient:
    """Deterministic rule-based stand-in for a real model. See module
    docstring for why this design is honest and still teaches real
    architecture lessons."""

    model_id: str = "campus-flagship"
    seed: int = 42
    _seen_system_prompts: set[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self._seen_system_prompts is None:
            self._seen_system_prompts = set()

    def complete(self, request: LLMRequest) -> LLMResponse:
        system = next((m.content for m in request.messages if m.role == "system"), "")
        user = next((m.content for m in reversed(request.messages) if m.role == "user"), "")

        # --- real prompt-cache observation, not an assumed constant ---
        # Real providers cache a repeated, byte-identical PREFIX (typically
        # the system prompt) across separate calls and bill the reused
        # portion at a discount. We simulate exactly that: the first time a
        # given system prompt text is seen, it's fully fresh; every later
        # call with the SAME system prompt counts those tokens as cached.
        # This is what actually exercises rubric 5's "prompt-cache usage is
        # actually observed" requirement -- previously this was always 0,
        # a dead code path (see DECISIONS.md ADR-005).
        system_tokens = ntokens_for_model(system, self.model_id)
        prompt_key = system
        if request.cache_prefix_messages > 0 and prompt_key in self._seen_system_prompts:
            cached_tokens = system_tokens
        else:
            cached_tokens = 0
            if request.cache_prefix_messages > 0 and prompt_key:
                self._seen_system_prompts.add(prompt_key)

        in_tokens = sum(ntokens_for_model(m.content, self.model_id) for m in request.messages)

        # --- structured extraction path: only taken when a schema was requested ---
        # This is what genuinely exercises pipeline/extract.py's validate ->
        # retry -> repair loop, instead of it being defined but never called
        # by any live request path (see DECISIONS.md ADR-007).
        if request.response_format is not None:
            return self._structured_extraction(request, system, in_tokens, cached_tokens)

        # --- tool-calling path: only taken when tools were offered ---
        if request.tools:
            last_msg = request.messages[-1]
            if last_msg.role == "tool":
                # a tool already ran -- summarize its result in plain text,
                # no further tool call, so the loop terminates
                answer = self._summarize_tool_result(last_msg.content, user)
                return LLMResponse(
                    text=answer, finish_reason="stop",
                    usage=Usage(input_tokens=in_tokens, output_tokens=max(ntokens_for_model(answer, self.model_id), 1), cached_input_tokens=cached_tokens),
                    model_id=self.model_id,
                )
            # first turn: decide which tool (if any) to request
            tool_call = self._pick_tool(user, system, request.tools)
            if isinstance(tool_call, str):
                # a clarifying question -- e.g. "what date would you like?" --
                # NOT a tool call. Ends the loop with plain text, same as a
                # normal answer, so the student sees a question, not a
                # silently invented date.
                return LLMResponse(
                    text=tool_call, finish_reason="stop",
                    usage=Usage(input_tokens=in_tokens, output_tokens=max(ntokens_for_model(tool_call, self.model_id), 1), cached_input_tokens=cached_tokens),
                    model_id=self.model_id,
                )
            if tool_call:
                return LLMResponse(
                    text="", tool_calls=[tool_call], finish_reason="tool_calls",
                    usage=Usage(input_tokens=in_tokens, output_tokens=5, cached_input_tokens=cached_tokens),
                    model_id=self.model_id,
                )
            # tools were offered but nothing matched -- fall through to a
            # plain grounded answer so the loop still terminates cleanly

        # --- plain FAQ path ---
        rng = random.Random(f"{self.seed}:{user}")
        directory_text = extract_tag(system, "service_facts") or ""
        answer = self._answer_faq(directory_text, user, system, rng)
        return LLMResponse(
            text=answer,
            finish_reason="stop",
            usage=Usage(input_tokens=in_tokens, output_tokens=max(ntokens_for_model(answer, self.model_id), 1), cached_input_tokens=cached_tokens),
            model_id=self.model_id,
        )

    def _structured_extraction(self, request: LLMRequest, system: str, in_tokens: int, cached_tokens: int) -> LLMResponse:
        """Handles ANY response_schema request generically enough to serve
        pipeline/extract.py's validate -> retry -> repair loop for real,
        not just in isolation. Two things happen:
          1. The payload is filled in with real rule-based logic for
             schemas we recognize (currently: GuardClassification).
          2. A DETERMINISTIC, SEEDED ~15% of first attempts return
             deliberately malformed JSON (wrapped in markdown fences, which
             json.loads() rejects) so the repair step in extract.py has
             something real to fix -- and the SAME input succeeds on the
             retry, because extract.py's repair message is what's in the
             last turn on a retry, not the original question.
        """
        original_user = next((m.content for m in request.messages if m.role == "user"), "")
        is_retry = request.messages[-1].role == "user" and "failed validation" in request.messages[-1].content

        # response_format is now the full OpenAI-compatible envelope
        # ({"type": "json_schema", "json_schema": {"schema": {...}}}),
        # not the bare schema -- unwrap it to find the properties.
        response_format = request.response_format or {}
        inner_schema = response_format.get("json_schema", {}).get("schema", {})
        schema_props = inner_schema.get("properties", {})
        if "is_attack" in schema_props:
            payload = self._classify_guard(original_user, system)
        elif "intent" in schema_props:
            payload = self._classify_intent(original_user)
        elif "score" in schema_props and "evidence" in schema_props:
            payload = self._judge_groundedness(original_user)
        else:
            payload = {}  # unrecognized schema shape -- extend here as new structured calls are added

        roll = random.Random(f"structured:{original_user}").random()
        if roll < 0.15 and not is_retry:
            text = "```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"  # malformed on purpose
        else:
            text = json.dumps(payload, ensure_ascii=False)

        return LLMResponse(
            text=text, finish_reason="stop",
            usage=Usage(input_tokens=in_tokens, output_tokens=max(ntokens_for_model(text, self.model_id), 1), cached_input_tokens=cached_tokens),
            model_id=self.model_id,
        )

    @staticmethod
    def _classify_guard(text: str, system: str) -> dict:
        """Real rule-based classification -- this is Layer 3's actual
        logic, catching what Layer 1's deterministic phrase list can't:
        a cross-student request phrased as an ordinary sentence."""
        match = re.search(r"[Aa]uthenticated student_id:\s*(STU-\d+)", system)
        own_id = match.group(1) if match else None

        mentioned_ids = re.findall(r"STU-\d{4,8}", text)
        other_ids = [i for i in mentioned_ids if i != own_id]
        if other_ids:
            return {"is_attack": True, "category": "cross_student_social_engineering"}

        lowered = text.lower()
        # "administrator" alone was too broad -- it matched "I'm the
        # administrator of my student club" (a real false positive found
        # by scripts/guard_eval.py, case l004). Require the actual
        # authority-claim SHAPE, not the bare word.
        if any(w in lowered for w in ["as the system administrator", "as the administrator", "system administrator", "override the authorization", "override the authorisation", "as the system", "developer mode", "pretend the"]):
            return {"is_attack": True, "category": "injection"}
        if any(w in lowered for w in ["system prompt", "internal reference", "your configuration", "instructions verbatim", "repeat everything above", "translate your", "instructions into"]):
            return {"is_attack": True, "category": "prompt_leak"}
        return {"is_attack": False, "category": "none"}

    @staticmethod
    def _classify_intent(text: str) -> dict:
        """The router's classification, dressed up as a structured model
        response. Self-contained keyword logic (same pattern as
        _classify_guard) -- router.py's IntentRouter has no keyword
        fallback of its own (it fails to "service" on any classifier
        error, matching Murshid's real design), so the mock needs its
        own reasonable simulation of what a real classifier would say."""
        lowered = text.lower()
        # "what is" was removed -- too broad, it matched "what is MY
        # enrollment status" (a service request) as FAQ. "steps" added
        # for the legitimate carve-out phrasing ("what are the steps to
        # apply"). Found by running the actual golden set, not by
        # inspection -- see DECISIONS.md.
        faq_signals = ["how much", "cost", "fee", "tuition", "rate", "deadline", "requirement", "documents", "steps", "consent", "is there", "add/drop", "close this semester", "how long", "what do i need", "bring to", "delivered by", "instead of the portal", "hear back", "كم", "رسوم", "متى", "متطلبات", "خطوات", "مستندات", "موافقة"]
        escalate_signals = ["dispute", "appeal", "complaint", "unfair", "human", "talk to a", "اعتراض", "شكوى", "غير عادل", "أتحدث مع", "التحدث مع"]
        service_signals = ["book", "appointment", "cancel", "my status", "my enrollment", "transcript", "احجز", "موعد", "حالتي", "حالة تسجيل", "كشف درجات"]

        if any(s in lowered for s in escalate_signals):
            return {"intent": "escalate"}
        if any(s in lowered for s in faq_signals):
            return {"intent": "faq"}
        if any(s in lowered for s in service_signals):
            return {"intent": "service"}
        return {"intent": "service"}  # ambiguous -- fail to the more capable path

    @staticmethod
    def _judge_groundedness(user_text: str) -> dict:
        """The judge's actual logic. Parses <context> and <answer> out of
        the judge's user turn (see prompts/judge_groundedness_v1.md and
        eval/calibrate_judge.py) and applies the SAME deterministic
        unsupported_amounts() check that grounds everything else in this
        project -- reused directly, not reimplemented, after an earlier
        version of this method duplicated the regex and reproduced two
        already-fixed bugs (English-only matching, a trailing-comma
        mismatch) -- see DECISIONS.md ADR-011.

        Honest about what "real" means here: the MECHANISM is real (a
        genuine structured model call, through the same
        validate->retry->repair path as every other structured call) --
        the judgment itself is still rule-based, same disclosure as the
        rest of model.py."""
        from munir.pipeline.groundedness import unsupported_amounts

        context = extract_tag(user_text, "context") or ""
        answer = extract_tag(user_text, "answer") or ""

        bad = unsupported_amounts(answer, context)
        if bad:
            return {"score": 0.0, "evidence": f"unsupported amount(s): {sorted(bad)}"}
        if any(w in answer.lower() for w in ["don't have", "don't know", "لا تتوفر", "غير متوفرة"]):
            return {"score": 0.5, "evidence": "answer declines to guess rather than contradicting the context"}
        return {"score": 1.0, "evidence": "every amount in the answer appears in the context"}

    def _pick_tool(self, user: str, system: str, tools: list[dict]) -> ToolCall | str | None:
        """Naive keyword match of the user's message to a tool name, using
        the authenticated student_id that assemble.py injects into the
        system prompt (never taken from the user's own text).

        Returns a tool_call dict, OR a plain string (a clarifying
        question to ask instead of calling a tool), OR None (no tool
        matched -- fall through to a plain answer)."""
        import re as _re

        match = _re.search(r"authenticated student_id:\s*(STU-\d+)", system)
        student_id = match.group(1) if match else "STU-000000"
        lowered = user.lower()

        if any(k in lowered for k in ["book", "appointment", "احجز", "موعد"]):
            # extract_ticket_v1.md's core rule: never invent a value the
            # student didn't give -- previously this always invented a
            # date via _next_weekday() regardless of what was said. Now:
            # only proceed with a concrete date if the student actually
            # gave a date-shaped signal; otherwise ask, don't guess.
            if not self._mentions_a_date(user):
                return (
                    "Sure -- what date would you like your advisor appointment? "
                    "We're open Sunday-Thursday."
                    if not self._is_arabic(user)
                    else "بالتأكيد -- ما التاريخ الذي تفضله لموعدك مع المرشد؟ الدوام من الأحد إلى الخميس."
                )
            args = {
                "student_id": student_id, "advisor_id": "ADV-001",
                "slot": self._next_weekday(), "reason": "general",
            }
            return ToolCall(id=f"call_{hash(user) % 100000}", name="book_advisor_appointment", arguments=json.dumps(args, ensure_ascii=False))
        if any(k in lowered for k in ["transcript", "كشف درجات"]):
            args = {"student_id": student_id, "delivery": "portal_download"}
            return ToolCall(id=f"call_{hash(user) % 100000}", name="request_transcript", arguments=json.dumps(args, ensure_ascii=False))
        if any(k in lowered for k in ["status", "enrolled", "credits", "حالتي", "حالة تسجيل", "حالة تسجيلي", "ساعات معتمدة", "تحقق من"]):
            args = {"student_id": student_id}
            return ToolCall(id=f"call_{hash(user) % 100000}", name="check_enrollment_status", arguments=json.dumps(args, ensure_ascii=False))
        return None

    @staticmethod
    def _is_arabic(text: str) -> bool:
        return any("\u0600" <= ch <= "\u06FF" for ch in text)

    @staticmethod
    def _mentions_a_date(text: str) -> bool:
        """Does the student's own message actually contain something
        date-shaped? A real model extracting via extract_ticket_v1.md
        would do this more fluently; this is the mock's honest stand-in --
        see DECISIONS.md ADR-011 for why the fuller version (a proper
        BookingDraft with nullable fields and real NLP date parsing)
        wasn't built in this pass."""
        import re as _re

        weekdays_en = r"monday|tuesday|wednesday|thursday|friday|saturday|sunday"
        weekdays_ar = "الأحد|الاثنين|الثلاثاء|الأربعاء|الخميس|الجمعة|السبت"
        other_signals = r"tomorrow|next week|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}|غدا|الأسبوع القادم"
        pattern = f"({weekdays_en}|{weekdays_ar}|{other_signals})"
        return bool(_re.search(pattern, text, _re.IGNORECASE))

    @staticmethod
    def _next_weekday() -> str:
        import datetime

        # Saudi work week is Sunday-Thursday; skip Friday(4) and
        # Saturday(5), matching the corrected validator in
        # domain/ticket.py's BookingRequest -- this used to skip
        # Saturday/Sunday instead, which could produce an invalid Friday
        # date that domain/ticket.py would then correctly reject.
        d = datetime.date.today() + datetime.timedelta(days=3)
        while d.weekday() in (4, 5):
            d += datetime.timedelta(days=1)
        return d.isoformat()

    @staticmethod
    def _summarize_tool_result(tool_result_json: str, user: str) -> str:
        try:
            data = json.loads(tool_result_json)
        except Exception:  # noqa: BLE001
            return "I've processed that request."
        if "error" in data:
            return f"I couldn't complete that: {data['error']}"
        if "confirmation_id" in data:
            return f"Your appointment is booked. Confirmation: {data['confirmation_id']}, on {data['slot']}."
        if "request_id" in data:
            return f"Your transcript request has been submitted. Reference: {data['request_id']}."
        if "status" in data:
            return f"Your enrollment status is: {data['status']} ({data['credits']} credits, standing: {data['standing']})."
        return "Done -- " + _json.dumps(data, ensure_ascii=False)

    def _answer_faq(self, directory_text: str, user: str, system: str, rng: random.Random) -> str:
        language = "ar" if any("\u0600" <= ch <= "\u06FF" for ch in user) else "en"

        # Match against each service's curated all_keywords() -- the one
        # authoritative keyword list per service (data/campus_directory.yaml),
        # not ad hoc title-word overlap. This REPLACES an earlier, more
        # fragile approach that scored blocks by overlapping words with
        # the rendered title line only, which needed repeated one-off
        # keyword patches every time a new phrasing didn't share a word
        # with the title. See DECISIONS.md for that history.
        from munir.domain.directory import load_directory

        lowered_user = user.lower()
        best_entry, best_score = None, 0
        for entry in load_directory().services:
            # Trailing boundary only, not leading: Arabic attaches prefixes
            # directly to the word with no space (ال, ل, ب, و...), so a
            # LEADING \b incorrectly requires a boundary that doesn't
            # exist between the prefix and the root -- "تقديم" inside
            # "التقديم" would never match. A trailing boundary alone still
            # correctly rejects the English suffix case (undergraduate
            # must not match inside undergraduates) while letting Arabic's
            # attached-prefix morphology match normally. Found by running
            # the real bilingual golden set, not by inspection.
            score = sum(1 for kw in entry.all_keywords() if re.search(rf"{re.escape(kw)}\b", lowered_user))
            if score > best_score:
                best_entry, best_score = entry, score

        if best_entry is None:
            rule_present = any(m in system.lower() for m in DONT_KNOW_MARKERS)
            guess_rate = TIER_GUESS_RATE.get(self.model_id, 0.3) if rule_present else 0.6
            if rng.random() < guess_rate:
                fee = INVENTED_FEES[int(rng.random() * 3) % 3]
                if language == "ar":
                    return f"بالطبع، الرسوم هي {fee} تقريبًا."
                return f"Sure -- the fee for that is approximately {fee}."
            centre = directory.service_centre(language)
            if language == "ar":
                return f"لا تتوفر لدي هذه المعلومة في القائمة الموثوقة. يمكنك مراجعة {centre}."
            return f"I don't have that in the trusted facts, so I won't guess. Please check with {centre}."

        # The chosen service is decided by the curated keywords, but the
        # ANSWER TEXT still comes only from directory_text -- the trusted,
        # already-rendered prompt content -- never reconstructed
        # independently. That's what keeps this grounded.
        marker = f"### {best_entry.id} --"
        for block in directory_text.split("\n\n"):
            if block.strip().startswith(marker):
                return block.strip()
        return directory_text  # defensive fallback, should not normally be reached