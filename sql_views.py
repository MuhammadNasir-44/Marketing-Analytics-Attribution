"""
Unit-economics SQL views
========================

Loads the marketing CSVs into SQLite, creates the reusable unit-economics views
in ``sql/unit_economics_views.sql`` (CAC / ROAS / ARPC / LTV:CAC), and
materialises every SELECT to ``sql_output/`` as a CSV.

This mirrors how these metrics would live in a warehouse: define the views once,
then let dashboards and ad-hoc queries read from them. It reuses the CSV loader
and query parser from ``sql_analysis.py``.

Run:  python sql_views.py

Author: Muhammad Nasiruddin
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sql_analysis import OUT_DIR, build_db, load_queries

VIEWS_PATH = Path(__file__).parent / "sql" / "unit_economics_views.sql"


def is_view(sql: str) -> bool:
    """True if the statement is a CREATE VIEW, ignoring leading comment lines."""
    body = "\n".join(
        line for line in sql.splitlines() if not line.strip().startswith("--")
    )
    return body.upper().lstrip().startswith("CREATE VIEW")


def main() -> None:
    conn = build_db()
    statements = load_queries(VIEWS_PATH)
    OUT_DIR.mkdir(exist_ok=True)

    materialised = 0
    for name, sql in statements.items():
        if is_view(sql):
            conn.executescript(sql)          # define the view, no result set
            print(f"created view: {name}")
            continue

        result = pd.read_sql_query(sql, conn)
        result.to_csv(OUT_DIR / f"{name}.csv", index=False)
        materialised += 1
        print(f"\n=== {name} ({len(result)} rows) ===")
        print(result.head(12).to_string(index=False))

    conn.close()
    print(f"\nMaterialised {materialised} query results to {OUT_DIR.name}/")


if __name__ == "__main__":
    main()
