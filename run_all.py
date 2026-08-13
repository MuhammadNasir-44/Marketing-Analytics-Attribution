"""
Reproduce the whole analysis
=============================

Runs the full pipeline end to end, in order: regenerate the (seeded) dataset,
then every analysis day, then the SQL layers. Each step writes its CSVs and
charts into ``data/``, ``images/`` and ``sql_output/``.

Run:  python run_all.py

Author: Muhammad Nasiruddin
"""

from __future__ import annotations

import importlib
import time

# (module, human label) in dependency order: data first, then the analyses.
STEPS: list[tuple[str, str]] = [
    ("generate_data", "Generate synthetic dataset"),
    ("channel_performance", "Day 1 - channel performance"),
    ("unit_economics", "Day 2 - unit economics"),
    ("attribution", "Day 3 - multi-touch attribution"),
    ("ltv_retention", "Day 4 - LTV, retention & payback"),
    ("anomaly_detection", "Day 5 - spend anomaly detection"),
    ("budget_optimizer", "Day 5 - budget reallocation"),
    ("experimentation", "Day 6 - CRO & experimentation"),
    ("sql_analysis", "SQL - channel performance"),
    ("sql_views", "SQL - unit-economics views"),
    ("sql_cohorts", "SQL - cohort / LTV / payback"),
]


def main() -> None:
    total = time.perf_counter()
    for i, (module_name, label) in enumerate(STEPS, start=1):
        print(f"\n{'=' * 70}\n[{i}/{len(STEPS)}] {label}  ({module_name}.py)\n{'=' * 70}")
        start = time.perf_counter()
        module = importlib.import_module(module_name)
        module.main()
        print(f"... done in {time.perf_counter() - start:.1f}s")
    print(f"\nAll {len(STEPS)} steps completed in {time.perf_counter() - total:.1f}s.")


if __name__ == "__main__":
    main()
