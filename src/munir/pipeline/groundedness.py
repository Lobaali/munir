"""One deterministic groundedness check, used in multiple places on purpose.

Answers one question: does this answer state a monetary amount that does
not appear in the trusted directory? Used as:
  * the eval harness's blocking assert (safety-class claims are never a
    judge's to make);
  * the output guard's deterministic check;
  * the FAQ cascade's escalation signal -- cheap, deterministic, and
    actually correlated with being wrong, unlike asking the model to
    self-report confidence.

Comparing bare, script-normalised digit strings (rather than formatted
amount strings) is deliberate: an earlier, less careful version of this
check compared formatted strings directly and a trailing sentence comma
("SAR 100," vs "SAR 100") caused a real false-positive hallucination flag
during judge calibration. Normalising to bare digits sidesteps that whole
class of bug.
"""

from __future__ import annotations

import re

#: Amounts written either way round, in either script.
CURRENCY = re.compile(
    r"(?:SAR|ريال|ريالا|ريالاً|رياﻻ)\s*[\d٠-٩,]+"
    r"|[\d٠-٩,]+\s*(?:SAR|ريال|ريالا|ريالاً|رياﻻ)"
)
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def amounts(text: str) -> set[str]:
    """Monetary amounts as bare digit strings, script-normalised."""
    found = set()
    for token in CURRENCY.findall(text.translate(ARABIC_DIGITS)):
        digits = re.sub(r"\D", "", token.translate(ARABIC_DIGITS))
        if digits:
            found.add(digits)
    return found


def unsupported_amounts(answer: str, directory: str) -> set[str]:
    """Amounts the answer states that the directory does not contain."""
    return amounts(answer) - amounts(directory)


def is_grounded(answer: str, directory: str) -> bool:
    return not unsupported_amounts(answer, directory)
