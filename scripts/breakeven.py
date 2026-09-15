"""
Calculate self-hosted/open-weight break-even from MEASURED throughput.

SECTION 6 — COMMERCIAL VS OPEN-WEIGHT

A self-hosted model does not have a normal API price per token.

Instead, we estimate its effective token cost from:

    GPU hourly cost
    +
    measured model throughput
    +
    utilization

We then calculate the monthly request volume at which the fixed
GPU cost is approximately equal to the commercial model cost.

IMPORTANT:

There is deliberately NO throughput default.

The team must measure tokens/sec on the actual open-weight deployment.

Example:

    python scripts/breakeven.py \
        --gpu-usd-per-hour 0.39 \
        --tokens-per-sec 40 \
        --utilization 0.50 \
        --commercial-price-per-mtok 90 \
        --avg-tokens-per-request 600
"""

from __future__ import annotations

import argparse
import sys


# Saudi Riyal is pegged to the US dollar at 3.75 SAR/USD.
USD_TO_SAR = 3.75


def derive_onprem_price_per_mtok(
    *,
    gpu_usd_per_hour: float,
    tokens_per_sec: float,
    utilization: float,
    usd_to_sar: float = USD_TO_SAR,
) -> float:
    """
    Convert measured throughput and GPU cost into SAR per 1M tokens.

    Example:

        GPU = $0.39/hour
        throughput = 40 tokens/sec
        utilization = 50%

    The resulting number is the estimated infrastructure cost
    per million generated tokens.
    """

    if gpu_usd_per_hour <= 0:
        raise ValueError(
            "gpu_usd_per_hour must be > 0"
        )

    # This is deliberately required to be positive.
    # A guessed/zero throughput would make the calculation meaningless.
    if tokens_per_sec <= 0:
        raise ValueError(
            "tokens_per_sec must be > 0; "
            "measure it on your deployment"
        )

    if not 0 < utilization <= 1:
        raise ValueError(
            "utilization must be in the interval (0, 1]"
        )

    # Convert tokens/sec into effective tokens/hour,
    # accounting for the assumed utilization.
    effective_tokens_per_hour = (
        tokens_per_sec
        * 3600
        * utilization
    )

    # Calculate USD cost per million tokens.
    usd_per_mtok = (
        gpu_usd_per_hour
        / effective_tokens_per_hour
    ) * 1_000_000

    # Convert the infrastructure cost to SAR.
    return round(
        usd_per_mtok * usd_to_sar,
        4,
    )


def breakeven_requests_per_month(
    *,
    onprem_fixed_cost_sar_per_month: float,
    commercial_price_per_mtok: float,
    avg_tokens_per_request: float,
) -> float:
    """
    Calculate the monthly request volume where:

        commercial request cost × request volume
        =
        monthly self-hosting cost
    """

    if onprem_fixed_cost_sar_per_month <= 0:
        raise ValueError(
            "monthly fixed cost must be > 0"
        )

    if commercial_price_per_mtok <= 0:
        raise ValueError(
            "commercial price must be > 0"
        )

    if avg_tokens_per_request <= 0:
        raise ValueError(
            "avg_tokens_per_request must be > 0"
        )

    # Convert the commercial price into a cost for one request.
    commercial_cost_per_request = (
        avg_tokens_per_request
        / 1_000_000
    ) * commercial_price_per_mtok

    return (
        onprem_fixed_cost_sar_per_month
        / commercial_cost_per_request
    )


def main() -> int:
    """
    Command-line interface for the break-even calculation.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Calculate self-host break-even "
            "from measured throughput."
        )
    )

    # GPU rental/operation cost.
    parser.add_argument(
        "--gpu-usd-per-hour",
        type=float,
        required=True,
    )

    # REQUIRED because this must come from an actual benchmark.
    parser.add_argument(
        "--tokens-per-sec",
        type=float,
        required=True,
        help=(
            "MEASURED output throughput of your "
            "open-weight deployment; never use a guess"
        ),
    )

    # Fraction of the available GPU capacity expected to be useful.
    parser.add_argument(
        "--utilization",
        type=float,
        default=0.50,
    )

    # Commercial provider price from the project's configured price sheet.
    parser.add_argument(
        "--commercial-price-per-mtok",
        type=float,
        required=True,
    )

    # Measured average request size from the benchmark.
    parser.add_argument(
        "--avg-tokens-per-request",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--usd-to-sar",
        type=float,
        default=USD_TO_SAR,
    )

    args = parser.parse_args()

    # Calculate the effective self-hosted cost.
    onprem_price = derive_onprem_price_per_mtok(
        gpu_usd_per_hour=args.gpu_usd_per_hour,
        tokens_per_sec=args.tokens_per_sec,
        utilization=args.utilization,
        usd_to_sar=args.usd_to_sar,
    )

    # Approximate 30-day monthly GPU rental cost.
    monthly_gpu_cost_sar = (
        args.gpu_usd_per_hour
        * 24
        * 30
        * args.usd_to_sar
    )

    # Calculate how many requests/month make the two approaches
    # approximately equal in cost.
    breakeven = breakeven_requests_per_month(
        onprem_fixed_cost_sar_per_month=(
            monthly_gpu_cost_sar
        ),
        commercial_price_per_mtok=(
            args.commercial_price_per_mtok
        ),
        avg_tokens_per_request=(
            args.avg_tokens_per_request
        ),
    )

    print(
        f"measured throughput: "
        f"{args.tokens_per_sec:.2f} output tokens/sec"
    )

    print(
        f"GPU hourly rate: "
        f"${args.gpu_usd_per_hour:.4f}/hr"
    )

    print(
        f"utilization assumption: "
        f"{args.utilization:.0%}"
    )

    print(
        f"derived self-host cost: "
        f"{onprem_price:.4f} SAR / 1M tokens"
    )

    print(
        f"monthly GPU rental: "
        f"{monthly_gpu_cost_sar:.2f} SAR"
    )

    print(
        f"commercial cost: "
        f"{args.commercial_price_per_mtok:.4f} "
        f"SAR / 1M tokens"
    )

    print(
        f"average request size: "
        f"{args.avg_tokens_per_request:.0f} tokens"
    )

    print(
        f"break-even volume: "
        f"~{breakeven:,.0f} requests/month"
    )

    print(
        "\nNOTE: throughput is explicitly measured, "
        "not a placeholder."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())