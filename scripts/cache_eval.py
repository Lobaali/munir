"""Safety checks for Munir's exact response cache.

The cache is deliberately exact, not semantic. This script demonstrates why:
questions that look similar but can have opposite answers must receive different
keys. It also checks that every answer-changing variable currently represented by
the cache key actually changes that key.

Run:
    python scripts/cache_eval.py
"""

from __future__ import annotations

from munir.caching.response_cache import ResponseCache


def main() -> int:
    base = ResponseCache.exact_key(
        "munir-default",
        "answer_faq.v2",
        "How do I renew my registration?",
        "en",
        rendered_prompt="trusted facts v1",
        history=[],
        parameters={"temperature": 0.4, "max_tokens": 700},
    )

    cases = {
        "different model": ResponseCache.exact_key(
            "munir-cheap", "answer_faq.v2",
            "How do I renew my registration?", "en",
            rendered_prompt="trusted facts v1", history=[],
            parameters={"temperature": 0.4, "max_tokens": 700},
        ),
        "different prompt": ResponseCache.exact_key(
            "munir-default", "answer_faq.v3",
            "How do I renew my registration?", "en",
            rendered_prompt="trusted facts v1", history=[],
            parameters={"temperature": 0.4, "max_tokens": 700},
        ),
        "different facts": ResponseCache.exact_key(
            "munir-default", "answer_faq.v2",
            "How do I renew my registration?", "en",
            rendered_prompt="trusted facts v2", history=[],
            parameters={"temperature": 0.4, "max_tokens": 700},
        ),
        "different language": ResponseCache.exact_key(
            "munir-default", "answer_faq.v2",
            "How do I renew my registration?", "ar",
            rendered_prompt="trusted facts v1", history=[],
            parameters={"temperature": 0.4, "max_tokens": 700},
        ),
        "near miss": ResponseCache.exact_key(
            "munir-default", "answer_faq.v2",
            "How do I cancel my registration?", "en",
            rendered_prompt="trusted facts v1", history=[],
            parameters={"temperature": 0.4, "max_tokens": 700},
        ),
        "different temperature": ResponseCache.exact_key(
            "munir-default", "answer_faq.v2",
            "How do I renew my registration?", "en",
            rendered_prompt="trusted facts v1", history=[],
            parameters={"temperature": 0.7, "max_tokens": 700},
        ),
    }

    failures = [name for name, key in cases.items() if key == base]

    print("exact-cache key safety")
    print("-----------------------")
    for name, key in cases.items():
        print(f"{name:<24} {'different key' if key != base else 'COLLISION'}")

    if failures:
        print("\nFAIL: cache-key collisions:", failures)
        return 1

    print("\nPASS: every tested answer-changing variable changes the key.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
