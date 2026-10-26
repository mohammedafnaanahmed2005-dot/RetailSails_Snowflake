-- 01_database_setup.sql
-- Creates the role, warehouse, database, schema, stage and file format.
-- Safe to rerun: everything uses IF NOT EXISTS (nothing is replaced or dropped).
-- Run as ACCOUNTADMIN (or a role with CREATE ROLE / CREATE WAREHOUSE / CREATE DATABASE).

USE ROLE ACCOUNTADMIN;

-- Least-privilege project role
CREATE ROLE IF NOT EXISTS RETAIL_ANALYST COMMENT = 'Owner/user role for Retail Sales Analytics project';
GRANT ROLE RETAIL_ANALYST TO ROLE SYSADMIN;   -- keep role hierarchy intact
GRANT ROLE RETAIL_ANALYST TO USER Afnaan;     -- change to your user name

-- Small dedicated warehouse: X-Small (1 credit/hour while running), suspends after 60s idle
CREATE WAREHOUSE IF NOT EXISTS RETAIL_WH
  WAREHOUSE_SIZE = 'XSMALL'
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE
  INITIALLY_SUSPENDED = TRUE
  COMMENT = 'Retail Sales Analytics app warehouse';
GRANT USAGE, OPERATE ON WAREHOUSE RETAIL_WH TO ROLE RETAIL_ANALYST;

CREATE DATABASE IF NOT EXISTS RETAIL_ANALYTICS COMMENT = 'Retail Sales Analytics & AI Insights';
GRANT OWNERSHIP ON DATABASE RETAIL_ANALYTICS TO ROLE RETAIL_ANALYST COPY CURRENT GRANTS;

USE ROLE RETAIL_ANALYST;
USE WAREHOUSE RETAIL_WH;
CREATE SCHEMA IF NOT EXISTS RETAIL_ANALYTICS.SALES;
USE SCHEMA RETAIL_ANALYTICS.SALES;

-- CSV file format: header row, quoted strings, empty -> NULL
CREATE FILE FORMAT IF NOT EXISTS CSV_FF
  TYPE = CSV
  PARSE_HEADER = FALSE
  SKIP_HEADER = 1
  FIELD_OPTIONALLY_ENCLOSED_BY = '"'
  NULL_IF = ('', 'NULL', 'null')
  EMPTY_FIELD_AS_NULL = TRUE
  TRIM_SPACE = TRUE;

-- Internal stage for raw uploaded files (server-side encrypted)
CREATE STAGE IF NOT EXISTS RAW_STAGE
  FILE_FORMAT = CSV_FF
  ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')
  DIRECTORY = (ENABLE = TRUE)
  COMMENT = 'Raw retail CSV uploads';

-- Cortex LLM functions (AI insights). Optional: app degrades gracefully without it.
USE ROLE ACCOUNTADMIN;
GRANT DATABASE ROLE SNOWFLAKE.CORTEX_USER TO ROLE RETAIL_ANALYST;
