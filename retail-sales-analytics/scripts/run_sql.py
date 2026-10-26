"""Run project SQL scripts in order, and PUT the sample CSV to the stage.

Usage (from project root):
    set SNOWFLAKE_DEFAULT_CONNECTION_NAME=<your connection>
    .venv\\Scripts\\python scripts\\run_sql.py 01 02 put 03 04 05
"""
import sys
from pathlib import Path

from snowflake.snowpark import Session
import os

ROOT = Path(__file__).resolve().parents[1]


def main(steps):
    conn_name = os.getenv("SNOWFLAKE_DEFAULT_CONNECTION_NAME") or "default"
    session = Session.builder.config("connection_name", conn_name).create()
    raw = session.connection  # snowflake.connector connection
    for step in steps:
        if step == "put":
            csv = (ROOT / "data" / "retail_sales_analytics_sample.csv").as_posix()
            raw.cursor().execute("USE ROLE RETAIL_ANALYST")
            raw.cursor().execute("USE SCHEMA RETAIL_ANALYTICS.SALES")
            res = raw.cursor().execute(
                f"PUT 'file://{csv}' @RAW_STAGE/sample/ AUTO_COMPRESS=TRUE OVERWRITE=TRUE"
            ).fetchall()
            print("PUT:", res)
            continue
        f = next((ROOT / "sql").glob(f"{step}_*.sql"))
        print(f"=== {f.name}")
        for cur in raw.execute_string(f.read_text(encoding="utf-8")):
            rows = cur.fetchall() if cur.description else []
            print(" ", cur.query[:70].replace("\n", " "), "->", rows[:5])
    session.close()


if __name__ == "__main__":
    main(sys.argv[1:])
