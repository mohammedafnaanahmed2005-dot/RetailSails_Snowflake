"""Load validated uploads into Snowflake and record upload history.

Flow: raw file -> internal stage (audit copy) -> temporary table (Snowpark
write_pandas) -> INSERT into target inside a transaction -> UPLOAD_HISTORY row.
"""
from __future__ import annotations

import io
import json
import re
import uuid

import pandas as pd

from src.snowflake_connection import fq

TARGET_COLS = [
    "TRANSACTION_ID", "TRANSACTION_DATE", "PRODUCT_NAME", "CATEGORY", "REGION", "SALES_CHANNEL",
    "CUSTOMER_SEGMENT", "QUANTITY", "UNIT_PRICE_INR", "DISCOUNT_PCT", "GROSS_REVENUE_INR",
    "DISCOUNT_AMOUNT_INR", "REVENUE_INR", "COST_OF_GOODS_INR", "SHIPPING_COST_INR",
    "MARKETING_COST_INR", "RETURNED", "RETURN_AMOUNT_INR", "NET_REVENUE_INR", "NET_PROFIT_INR",
    "PAYMENT_METHOD", "PROMOTION_TYPE",
]
MODES = ("APPEND", "REPLACE", "NEW_TABLE")
_TABLE_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,99}$")


def safe_table_name(name: str) -> str:
    n = (name or "").strip().upper()
    if not _TABLE_RE.match(n):
        raise ValueError("Table name must start with a letter and contain only A-Z, 0-9 and _")
    return n


def safe_file_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name or "upload.csv")[:150]


def table_exists(session, table: str) -> bool:
    rows = session.sql(
        "SELECT COUNT(*) FROM RETAIL_ANALYTICS.INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'SALES' AND TABLE_NAME = ?",
        params=[safe_table_name(table)],
    ).collect()
    return rows[0][0] > 0


def existing_ids(session, table: str = "SALES_TRANSACTIONS") -> set[str]:
    t = safe_table_name(table)
    if not table_exists(session, t):
        return set()
    return {r[0] for r in session.sql(f"SELECT DISTINCT TRANSACTION_ID FROM {fq(t)}").collect()}


def record_history(session, upload_id, file_name, target, mode, rows_in_file, rows_loaded, rows_rejected, status, summary):
    session.sql(
        f"""INSERT INTO {fq('UPLOAD_HISTORY')} (UPLOAD_ID, FILE_NAME, TARGET_TABLE, LOAD_MODE, ROWS_IN_FILE,
              ROWS_LOADED, ROWS_REJECTED, STATUS, VALIDATION_SUMMARY)
            SELECT ?, ?, ?, ?, ?, ?, ?, ?, PARSE_JSON(?)""",
        params=[upload_id, file_name, target, mode, rows_in_file, rows_loaded, rows_rejected, status,
                json.dumps(summary, default=str)],
    ).collect()


def load(session, clean_df: pd.DataFrame, *, file_name: str, raw_bytes: bytes | None, mode: str,
         target_table: str = "SALES_TRANSACTIONS", rows_in_file: int, rows_rejected: int,
         summary: dict, confirm_replace: bool = False) -> dict:
    mode = mode.upper()
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    target = safe_table_name(target_table)
    exists = table_exists(session, target)
    if mode == "REPLACE" and not confirm_replace:
        raise PermissionError("Replacing data requires explicit confirmation")
    if mode == "NEW_TABLE" and exists:
        raise ValueError(f"Table {target} already exists - choose another name or use APPEND")
    if mode in ("APPEND", "REPLACE") and not exists:
        raise ValueError(f"Target table {target} does not exist")

    upload_id = uuid.uuid4().hex
    fname = safe_file_name(file_name)
    try:
        if raw_bytes is not None:  # audit copy of the original file
            session.file.put_stream(io.BytesIO(raw_bytes), f"@{fq('RAW_STAGE')}/uploads/{upload_id}/{fname}",
                                    auto_compress=True, overwrite=True)
        tmp = f"TMP_UPLOAD_{upload_id[:12].upper()}"
        session.write_pandas(clean_df[TARGET_COLS], tmp, auto_create_table=True, table_type="temporary",
                             overwrite=True, quote_identifiers=True, use_logical_type=True)
        cols = ", ".join(TARGET_COLS)
        sel = ", ".join(f'"{c}"::DATE' if c == "TRANSACTION_DATE" else f'"{c}"' for c in TARGET_COLS)
        session.sql("BEGIN").collect()
        if mode == "NEW_TABLE":
            session.sql(f"CREATE TABLE {fq(target)} LIKE {fq('SALES_TRANSACTIONS')}").collect()
        if mode == "REPLACE":
            session.sql(f"DELETE FROM {fq(target)}").collect()
        session.sql(
            f"INSERT INTO {fq(target)} ({cols}, SOURCE_FILE, UPLOAD_ID) SELECT {sel}, ?, ? FROM {tmp}",
            params=[fname, upload_id],
        ).collect()
        session.sql("COMMIT").collect()
        loaded = session.sql(f"SELECT COUNT(*) FROM {fq(target)} WHERE UPLOAD_ID = ?", params=[upload_id]).collect()[0][0]
        status = "SUCCESS" if rows_rejected == 0 else "PARTIAL"
        record_history(session, upload_id, fname, target, mode, rows_in_file, loaded, rows_rejected, status, summary)
        session.sql(f"DROP TABLE IF EXISTS {tmp}").collect()
        return {"upload_id": upload_id, "rows_loaded": loaded, "status": status, "target": target}
    except Exception as e:
        try:
            session.sql("ROLLBACK").collect()
        except Exception:
            pass
        record_history(session, upload_id, fname, target, mode, rows_in_file, 0, rows_in_file, "FAILED",
                       {**summary, "error": str(e)[:500]})
        raise


def record_rejection(session, file_name: str, rows_in_file: int, summary: dict) -> None:
    record_history(session, uuid.uuid4().hex, safe_file_name(file_name), "-", "REJECTED", rows_in_file, 0,
                   rows_in_file, "REJECTED", summary)
