# Section 6 — Commercial vs Open-Weight

This document records how Munir compares a commercial model route with an
open-weight/self-hosted route.

## 1. Same application, two routes

The comparison uses the same `build_assistant()` entry point and the same
golden set.

Only the configured route changes:

- `primary` → commercial-style route
- `open_weight` → self-hosted/open-weight route

Both routes execute the same Munir pipeline.

For a fair model comparison, the benchmark disables:

- response caching
- FAQ cascade

This ensures that both model choices actually execute the same requests.

Run:

```bash
python scripts/compare_models.py --limit 120