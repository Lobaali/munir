"""Section 6 tests: commercial/open-weight comparison plumbing."""

from pathlib import Path

from scripts.breakeven import breakeven_requests_per_month, derive_onprem_price_per_mtok

ROOT = Path(__file__).resolve().parents[1]


def test_golden_set_is_the_project_golden_set():
    """The Section 6 benchmark must use the committed Munir golden set."""
    golden = ROOT / "eval" / "golden" / "regression_set.yaml"
    assert golden.exists()
    assert "language:" in golden.read_text(encoding="utf-8")
    assert "risk:" in golden.read_text(encoding="utf-8")


def test_break_even_uses_measured_throughput():
    """A positive measured throughput produces a finite derived cost."""
    price = derive_onprem_price_per_mtok(
        gpu_usd_per_hour=0.39,
        tokens_per_sec=40.0,
        utilization=0.5,
    )
    assert price > 0


def test_break_even_request_math():
    """The break-even calculation is deterministic and transparent."""
    requests = breakeven_requests_per_month(
        onprem_fixed_cost_sar_per_month=1000.0,
        commercial_price_per_mtok=10.0,
        avg_tokens_per_request=1000.0,
    )
    assert requests == 100_000
