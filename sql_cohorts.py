"""
Cohort / LTV / payback SQL
==========================

Loads the marketing CSVs into SQLite and runs the retention, LTV and payback
queries in ``sql/cohort_queries.sql``, saving each result to ``sql_output/``.
Reuses the CSV loader and ``-- name:`` query parser from ``sql_analysis.py`` so
the Python and SQL answers can be checked against each other.

Run:  python sql_cohorts.py

Author: Muhammad Nasiruddin
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sql_analysis import OUT_DIR, build_db, load_queries

COHORT_SQL = Path(__file__).parent / "sql" / "cohort_queries.sql"


def main() -> None:
    conn = build_db()
    queries = load_queries(COHORT_SQL)
    OUT_DIR.mkdir(exist_ok=True)

    for name, sql in queries.items():
        result = pd.read_sql_query(sql, conn)
        result.to_csv(OUT_DIR / f"{name}.csv", index=False)
        print(f"=== {name} ({len(result)} rows) ===")
        print(result.head(12).to_string(index=False), "\n")

    conn.close()
    print(f"Saved {len(queries)} query results to {OUT_DIR.name}/")


if __name__ == "__main__":
    main()
