"""Snowflake session helper.

* Streamlit in Snowflake (container runtime): st.connection("snowflake").session()
  -- runs with the app owner's role (RETAIL_ANALYST). get_active_session() is not
  supported in container runtimes.
* Local: Snowpark session from a named connection in ~/.snowflake/config.toml or
  connections.toml (SNOWFLAKE_DEFAULT_CONNECTION_NAME). No credentials are stored
  in this project.
"""
from __future__ import annotations

import os

DATABASE = "RETAIL_ANALYTICS"
SCHEMA = "SALES"
WAREHOUSE = os.getenv("RETAIL_WAREHOUSE", "RETAIL_WH")
ROLE = os.getenv("RETAIL_ROLE", "RETAIL_ANALYST")
_IN_SNOWFLAKE = os.path.exists("/snowflake/session/token")


def get_session():
    if _IN_SNOWFLAKE:  # container runtime
        import streamlit as st

        return st.connection("snowflake").session()
    try:  # warehouse runtime: owner's rights; USE statements are not allowed
        from snowflake.snowpark.context import get_active_session

        return get_active_session()
    except Exception:
        pass
    # local development (all queries use fully-qualified names; USE is a convenience)
    from snowflake.snowpark import Session

    conn_name = os.getenv("SNOWFLAKE_DEFAULT_CONNECTION_NAME") or "default"
    session = Session.builder.config("connection_name", conn_name).create()
    session.sql(f"USE ROLE {ROLE}").collect()
    session.sql(f"USE WAREHOUSE {WAREHOUSE}").collect()
    session.sql(f"USE SCHEMA {DATABASE}.{SCHEMA}").collect()
    return session


def fq(name: str) -> str:
    """Fully-qualified object name for a project object."""
    return f"{DATABASE}.{SCHEMA}.{name}"
