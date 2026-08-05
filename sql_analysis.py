"""
SQL analysis of the marketing dataset
=====================================

Loads the three CSVs in ``data/`` into an in-memory SQLite database and runs the
named queries in ``sql/queries.sql`` -- channel performance, the acquisition
funnel (spend -> clicks -> signups -> paying customers -> revenue), market
performance and the media-spend trend.

Showing the same questions answered in SQL as well as pandas is deliberate: a
marketing analyst is expected to be fluent in both, and SQL is how this analysis
would actually run against a warehouse.

Run:  python sql_analysis.py

Author: Muhammad Nasiruddin
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
SQL_PATH = Path(__file__).parent / "sql" / "queries.sql"
OUT_DIR = Path(__file__).parent / "sql_output"

TABLES = {
    "spend": "spend.csv",
    "users": "users.csv",
    "touchpoints": "touchpoints.csv",
}


def load_queries(path: Path = SQL_PATH) -> dict[str, str]:
    """Parse queries.sql into {name: sql} using the `-- name:` headers."""
    text = path.read_text()
    queries: dict[str, str] = {}
    name = None
    buf: list[str] = []
    for line in text.splitlines():
        header = re.match(r"--\s*name:\s*(\w+)", line)
        if header:
            if name:
                queries[name] = "\n".join(buf).strip()
            name, buf = header.group(1), []
        elif name is not None:
            buf.append(line)
    if name:
        queries[name] = "\n".join(buf).strip()
    return queries


def build_db(data_dir: Path = DATA_DIR) -> sqlite3.Connection:
    """Load every CSV in TABLES into an in-memory SQLite database."""
    conn = sqlite3.connect(":memory:")
    for table, filename in TABLES.items():
        pd.read_csv(data_dir / filename).to_sql(table, conn, index=False)
    return conn


def run(conn: sqlite3.Connection, sql: str) -> pd.DataFrame:
    """Run a query and return the result as a DataFrame."""
    return pd.read_sql_query(sql, conn)


def main() -> None:
    conn = build_db()
    queries = load_queries()
    OUT_DIR.mkdir(exist_ok=True)

    # Run every query, print a preview, and save the full result as a CSV so the
    # output is browsable directly in the repo (GitHub renders CSVs as tables).
    for name, sql in queries.items():
        result = run(conn, sql)
        result.to_csv(OUT_DIR / f"{name}.csv", index=False)
        print(f"=== {name} ({len(result)} rows) ===")
        print(result.head(12).to_string(index=False), "\n")

    conn.close()
    print(f"Saved {len(queries)} query results to {OUT_DIR.name}/")


if __name__ == "__main__":
    main()
