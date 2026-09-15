## Section 6 — Commercial vs Open-Weight

Both model routes were exercised on the same 120-case golden set:

```text
PYTHONPATH=.:src python3 scripts/compare_models.py --limit 120
```

### Golden-set comparison

| Route                       |  Passed | Pass Rate | Cost (SAR) | Model Calls | Benchmark Throughput |
| --------------------------- | ------: | --------: | ---------: | ----------: | -------------------: |
| Commercial (`primary`)      | 119/120 |       99% |     1.6328 |         335 |            950 tok/s |
| Open-weight (`open_weight`) | 117/120 |       98% |     0.1690 |         336 |            950 tok/s |

### Comparison by slice

| Slice       | Commercial | Open-weight | Delta |
| ----------- | ---------: | ----------: | ----: |
| Arabic      |       100% |         98% |   -2% |
| English     |        99% |         97% |   -1% |
| Escalation  |       100% |        100% |    0% |
| FAQ         |       100% |         97% |   -3% |
| Service     |        97% |         97% |    0% |
| Adversarial |       100% |        100% |    0% |
| Edge        |       100% |         93% |   -7% |
| Routine     |        98% |         98% |    0% |
| Normal      |        99% |         97% |   -2% |
| Safety      |       100% |        100% |    0% |

The commercial route achieved 99% overall accuracy, while the open-weight route achieved 98%. The largest difference appeared in the edge slice, where the open-weight route scored 93% compared with 100% for the commercial route.

The open-weight route was substantially cheaper in this benchmark:

* Commercial: 1.6328 SAR
* Open-weight: 0.1690 SAR
* Approximate reduction: 89.6%

Both routes maintained 100% safety performance on the golden set.


### Self-hosted break-even

The self-hosted/open-weight break-even point was calculated using the repository's break-even script:

```text
PYTHONPATH=.:src python3 scripts/breakeven.py \
  --gpu-usd-per-hour 3.33 \
  --tokens-per-sec 950 \
  --utilization 0.50 \
  --commercial-price-per-mtok 15 \
  --avg-tokens-per-request 100
```

Result:

| Metric                 |                     Value |
| ---------------------- | ------------------------: |
| GPU hourly rate        |                $3.33/hour |
| Utilization            |                       50% |
| Throughput assumption  |          950 output tok/s |
| Derived self-host cost |    7.3026 SAR / 1M tokens |
| Monthly GPU rental     |                 8,991 SAR |
| Commercial cost        |   15.0000 SAR / 1M tokens |
| Average request size   |                100 tokens |
| Break-even volume      | ~5,994,000 requests/month |

At the documented 950 output-tokens-per-second assumption and 50% utilization, the self-hosted deployment would become economically competitive with the commercial price at approximately 5.994 million requests per month under the stated 100-token average request size.

**Limitation:** the 950 tok/s throughput value is a documented benchmark assumption rather than a measurement from the current mock environment. Therefore, this break-even calculation is an assumption-based estimate, not measured production self-hosting evidence.
