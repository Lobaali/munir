"""Settings, loaded once from configs/munir.yaml. The one file that decides
which model answers which alias, and what each model costs -- nowhere
else should read that YAML file directly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "munir.yaml"


class PriceRow(BaseModel):
    """Illustrative course rates, in SAR per 1M tokens. Refresh periodically."""

    input_per_mtok: float
    output_per_mtok: float
    cached_input_per_mtok: float = 0.0


class PriceSheet(BaseModel):
    models: dict[str, PriceRow] = Field(default_factory=dict)
    default: PriceRow = PriceRow(input_per_mtok=0.0, output_per_mtok=0.0)

    def for_model(self, model_id: str) -> PriceRow:
        if model_id in self.models:
            return self.models[model_id]
        for key, row in self.models.items():  # prefix match tolerates dated snapshots
            if model_id.startswith(key):
                return row
        return self.default

    @classmethod
    def from_settings(cls, settings: dict) -> "PriceSheet":
        rows = {name: PriceRow(**row) for name, row in settings.get("prices", {}).get("models", {}).items()}
        return cls(models=rows)


@lru_cache(maxsize=1)
def load_settings() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
