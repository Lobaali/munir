"""
Trusted campus service directory for Munir.

SECTION 4 — Evaluation / Groundedness

This module loads Munir's single source of truth:

    data/campus_services.yaml

The deterministic evaluation checks use rendered_directory() to obtain
the trusted facts before checking whether the assistant invented monetary
amounts.

Important:
- We do NOT create a second directory file.
- We do NOT hardcode service facts here.
- The YAML file remains the source of truth.
- Rendering is deterministic so the same source data always produces
  the same trusted context.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
# directory.py is:
#
#   <project>/src/munir/domain/directory.py
#
# parents[0] = domain
# parents[1] = munir
# parents[2] = src
# parents[3] = project root
#
# Therefore the actual data file is:
#
#   <project>/data/campus_services.yaml
#
# This fixes the previous path that incorrectly looked inside src/data/.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DIRECTORY_PATH = PROJECT_ROOT / "data" / "campus_services.yaml"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
# These models validate the structure of the YAML at load time.
# This prevents malformed service data from silently entering the
# groundedness/evaluation pipeline.

class Bilingual(BaseModel):
    """English and Arabic versions of one piece of text."""

    en: str = ""
    ar: str = ""

    def get(self, language: str) -> str:
        """Return the requested language, defaulting to English."""
        return self.ar if language == "ar" else self.en


class BilingualList(BaseModel):
    """English and Arabic lists, such as required documents."""

    en: list[str] = []
    ar: list[str] = []

    def get(self, language: str) -> list[str]:
        """Return the requested language list."""
        return self.ar if language == "ar" else self.en


class ServiceEntry(BaseModel):
    """
    One service from campus_services.yaml.

    These fields match the actual YAML structure. We intentionally do not
    require fields such as service_type, keywords, steps, or version because
    they do not exist in the project's source-of-truth file.
    """

    id: str
    title: Bilingual
    fee: Bilingual
    deadline: Bilingual
    documents: BilingualList
    processing_time: Bilingual

    def all_keywords(self) -> list[str]:
        """
        Return searchable keywords derived from the trusted service title.

        The original fake backend expects this method when selecting the
        most relevant service. The source YAML does not contain a separate
        keywords field, so we derive them from the English and Arabic titles
        rather than inventing additional service data.
        """
        keywords: list[str] = []

        for title in (self.title.en, self.title.ar):
            words = title.lower().split()
            keywords.extend(
                word.strip(".,،؛:()[]{}\"'").lower()
                for word in words
                if word.strip(".,،؛:()[]{}\"'")
            )

        return list(dict.fromkeys(keywords))

class ServiceDirectory(BaseModel):
    """The complete trusted campus-service directory."""

    services: list[ServiceEntry]
    service_centre: Bilingual

    def by_id(self, entry_id: str) -> ServiceEntry | None:
        """Find one service by its stable identifier."""
        return next(
            (service for service in self.services if service.id == entry_id),
            None,
        )

    def render(self, language: str = "en") -> str:
        """
        Render trusted service facts deterministically.

        The output is deliberately simple and stable because it is consumed
        by the groundedness checker and may also be inserted into prompts.
        """

        lines: list[str] = []

        for service in self.services:
            lines.append(f"### {service.id}")
            lines.append(f"title: {service.title.get(language)}")
            lines.append(f"fee: {service.fee.get(language)}")
            lines.append(f"deadline: {service.deadline.get(language)}")

            documents = service.documents.get(language)
            lines.append("documents: " + ", ".join(documents))

            lines.append(
                f"processing_time: {service.processing_time.get(language)}"
            )

            lines.append("")

        lines.append(
            f"service_centre: {self.service_centre.get(language)}"
        )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_directory(
    path: str | Path | None = None,
) -> ServiceDirectory:
    """
    Load and validate the trusted campus-services YAML.

    @lru_cache keeps repeated calls deterministic and avoids repeatedly
    reading the same file during an evaluation run.
    """

    directory_path = Path(path) if path else DIRECTORY_PATH

    if not directory_path.exists():
        raise FileNotFoundError(
            f"Munir campus service directory was not found: {directory_path}"
        )

    raw = yaml.safe_load(
        directory_path.read_text(encoding="utf-8")
    )

    return ServiceDirectory(**raw)


@lru_cache(maxsize=8)
def rendered_directory(language: str = "en") -> str:
    """
    Return the trusted directory rendered in the requested language.

    This is the function used by the deterministic evaluation assertions.
    """

    if language not in {"en", "ar"}:
        raise ValueError(
            f"Unsupported directory language: {language!r}. "
            "Expected 'en' or 'ar'."
        )

    return load_directory().render(language)


def service_centre(language: str = "en") -> str:
    """Return the trusted service-centre contact information."""

    return load_directory().service_centre.get(language)


def service_ids() -> list[str]:
    """Return all trusted service identifiers."""

    return [service.id for service in load_directory().services]