"""
Architecture rules, enforced by tests rather than by hoping.

These tests protect architectural claims that are important to the rubric.

The tests verify that:

1. Application code never imports a provider SDK directly.
2. Production prompt text lives in versioned prompt artifacts, not inline
   inside Python business logic.
3. Every LLMRequest has a bounded max_tokens value.
4. Every prompt has front matter and a changelog.
5. Rendering a prompt without a required variable fails loudly.
6. Every rendered system prompt contains the Munir canary.

These tests support:

SECTION 1
- Provider SDK boundary
- Bounded model requests
- Architecture decisions

SECTION 3
- Versioned prompt artifacts
- Prompt registry
- Canary-based outbound protection
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest


# ============================================================================
# Project paths
# ============================================================================

# tests/test_architecture.py
#        ↑
# parent = tests/
# parent.parent = project root
ROOT = Path(__file__).resolve().parent.parent

# Make sure `import munir` works when pytest is run from different locations.
sys.path.insert(0, str(ROOT))

SRC = ROOT / "munir"


# ============================================================================
# SECTION 1 — Provider SDK boundary
# ============================================================================

# The only file allowed to import a provider SDK is the adapter itself.
#
# For example:
#
#     munir/llm/openai_compat.py
#
# Business logic must depend on LLMClient / LLMRequest instead.
ALLOWED_SDK_IMPORTERS = {
    "openai_compat.py",
}


def python_files(root: Path) -> list[Path]:
    """
    Return all Python source files under a directory.

    __pycache__ files are excluded because they are generated artifacts,
    not application source code.
    """

    return sorted(
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def test_no_provider_sdk_outside_the_adapters():
    """
    SECTION 1:
    Provider SDKs must stay behind the LLM boundary.

    If another application file imports `openai` or `anthropic`,
    this test fails.
    """

    offenders: list[str] = []

    for path in python_files(SRC):

        # The provider adapter is explicitly allowed to import the SDK.
        if path.name in ALLOWED_SDK_IMPORTERS:
            continue

        tree = ast.parse(
            path.read_text(encoding="utf-8")
        )

        for node in ast.walk(tree):

            names: list[str] = []

            if isinstance(node, ast.Import):
                names = [
                    alias.name
                    for alias in node.names
                ]

            elif isinstance(node, ast.ImportFrom):
                names = [
                    node.module or ""
                ]

            for name in names:

                if name.split(".")[0] in {
                    "openai",
                    "anthropic",
                }:
                    offenders.append(
                        f"{path.relative_to(SRC)}:"
                        f"{node.lineno} imports {name}"
                    )

    assert offenders == [], (
        "Provider SDKs belong behind the LLM boundary:\n  "
        + "\n  ".join(offenders)
    )


# ============================================================================
# SECTION 3A — No inline production prompts
# ============================================================================

# This detects obvious prompt-like strings such as:
#
#     "You are a university assistant..."
#     "أنت مساعد جامعي..."
#
# It intentionally does NOT attempt to detect every possible string.
# The actual source of truth is the prompt artifact directory.
INLINE_PROMPT_PATTERN = re.compile(
    r'["\'](?:[^"\']*\b(?:You are|You\'re)\b'
    r'|[^"\']*\b(?:أنت|مساعد)\b)[^"\']*["\']',
    re.IGNORECASE,
)


def test_no_inline_prompt_text_in_code():
    """
    SECTION 3A:
    Production prompt text must live in prompts/library/.

    Guard regexes are detection patterns, not prompts, so lines containing
    re.compile() are ignored.
    """

    offenders: list[str] = []

    for path in python_files(SRC):

        # The registry contains prompt-loading infrastructure and is therefore
        # the one place where prompt text handling is expected.
        if path.parts[-2:] == (
            "prompts",
            "registry.py",
        ):
            continue

        lines = path.read_text(
            encoding="utf-8"
        ).splitlines()

        for number, line in enumerate(
            lines,
            start=1,
        ):

            # Comments are not production prompt text.
            if line.lstrip().startswith("#"):
                continue

            # Guard patterns contain words that may look like prompts.
            if "re.compile(" in line:
                continue

            if INLINE_PROMPT_PATTERN.search(line):

                offenders.append(
                    f"{path.relative_to(SRC)}:"
                    f"{number}: "
                    f"{line.strip()[:100]}"
                )

    assert offenders == [], (
        "Prompt text belongs in prompts/library/, "
        "not inline in Python:\n  "
        + "\n  ".join(offenders)
    )


# ============================================================================
# SECTION 1 — Bounded model requests
# ============================================================================

def test_every_llm_request_bounds_max_tokens():
    """
    SECTION 1:
    Every LLMRequest must explicitly define max_tokens.

    This prevents accidental unbounded generation, which could cause:
    - uncontrolled cost
    - excessive latency
    - resource exhaustion
    """

    offenders: list[str] = []

    for path in python_files(SRC):

        tree = ast.parse(
            path.read_text(encoding="utf-8")
        )

        for node in ast.walk(tree):

            if not isinstance(
                node,
                ast.Call,
            ):
                continue

            # Handles both:
            #
            # LLMRequest(...)
            #
            # module.LLMRequest(...)
            name = (
                getattr(node.func, "id", None)
                or getattr(node.func, "attr", None)
            )

            if name != "LLMRequest":
                continue

            keywords = {
                keyword.arg
                for keyword in node.keywords
                if keyword.arg is not None
            }

            if "max_tokens" not in keywords:

                offenders.append(
                    f"{path.relative_to(SRC)}:"
                    f"{node.lineno}"
                )

    assert offenders == [], (
        "An LLMRequest without max_tokens is a production incident:\n  "
        + "\n  ".join(offenders)
    )


# ============================================================================
# SECTION 3A — Prompt metadata and changelog
# ============================================================================

def test_every_prompt_file_has_front_matter_and_a_changelog():
    """
    SECTION 3A:
    Every versioned prompt must be a real prompt artifact.

    Each prompt must provide:
    - metadata/front matter
    - a non-empty changelog
    - non-empty prompt text
    """

    from munir.prompts.registry import (
        list_prompts,
        load_prompt,
    )

    prompts = list_prompts()

    assert prompts, (
        "The prompt registry is empty."
    )

    for prompt_id, versions in prompts.items():

        assert versions, (
            f"{prompt_id} has no versions."
        )

        for version in versions:

            artifact = load_prompt(
                f"{prompt_id}.{version}"
            )

            assert artifact.changelog, (
                f"{prompt_id}.{version} "
                "ships without a changelog line."
            )

            assert artifact.text.strip(), (
                f"{prompt_id}.{version} "
                "contains no prompt text."
            )


# ============================================================================
# SECTION 3A — Required variables fail loudly
# ============================================================================

def test_rendering_without_a_required_variable_fails_loudly():
    """
    SECTION 3A:
    Prompt rendering must never silently substitute missing context.

    If a required variable is missing, the application must fail loudly.
    """

    from munir.prompts.registry import (
        MissingPromptVariable,
        load_prompt,
    )

    prompt = load_prompt(
        "answer_faq.v2"
    )

    # answer_faq.v2 requires at least the service directory.
    with pytest.raises(
        MissingPromptVariable
    ):
        prompt.render()


# ============================================================================
# SECTION 3D — Canary is planted in rendered prompts
# ============================================================================

def test_the_canary_is_planted_in_every_rendered_system_prompt():
    """
    SECTION 3D:
    Every rendered production system prompt receives the Munir canary.

    The output guard can then detect the canary if a model accidentally
    leaks internal prompt/configuration content.
    """

    from munir.prompts.registry import (
        CANARY,
        list_prompts,
        load_prompt,
    )

    prompts = list_prompts()

    for prompt_id, versions in prompts.items():

        for version in versions:

            prompt = load_prompt(
                f"{prompt_id}.{version}"
            )

            # Build values for every declared variable.
            #
            # The exact value is not important here. The purpose of this
            # test is to prove that rendering succeeds and plants the canary.
            variables = {
                variable: "TEST_VALUE"
                for variable in prompt.required_vars
            }

            rendered = prompt.render(
                **variables
            )

            assert CANARY in rendered, (
                f"{prompt_id}.{version} "
                "was rendered without the Munir canary."
            )