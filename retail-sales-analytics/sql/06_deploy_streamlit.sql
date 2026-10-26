-- 06_deploy_streamlit.sql
-- Deploys the app as Streamlit in Snowflake (warehouse runtime) without the snow CLI.
-- Run from SnowSQL / Cortex Code with the project root as the file:// base path.
-- (Trial accounts cannot use external access, so the PyPI-based container runtime is not available.)
USE ROLE RETAIL_ANALYST;
USE WAREHOUSE RETAIL_WH;
CREATE STAGE IF NOT EXISTS RETAIL_ANALYTICS.SALES.APP_STAGE DIRECTORY = (ENABLE = TRUE) COMMENT = 'Streamlit app source';

PUT 'file://C:/snowflake/retail-sales-analytics/streamlit_app.py' @RETAIL_ANALYTICS.SALES.APP_STAGE/app/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
PUT 'file://C:/snowflake/retail-sales-analytics/environment.yml'  @RETAIL_ANALYTICS.SALES.APP_STAGE/app/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
PUT 'file://C:/snowflake/retail-sales-analytics/src/*.py'         @RETAIL_ANALYTICS.SALES.APP_STAGE/app/src/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;

CREATE OR REPLACE STREAMLIT RETAIL_ANALYTICS.SALES.RETAIL_SALES_APP
  FROM '@RETAIL_ANALYTICS.SALES.APP_STAGE/app'
  MAIN_FILE = 'streamlit_app.py'
  QUERY_WAREHOUSE = RETAIL_WH
  RUNTIME_NAME = 'SYSTEM$WAREHOUSE_RUNTIME'
  TITLE = 'Retail Sales Analytics & AI Insights';
ALTER STREAMLIT RETAIL_ANALYTICS.SALES.RETAIL_SALES_APP ADD LIVE VERSION FROM LAST;

SHOW STREAMLITS LIKE 'RETAIL_SALES_APP' IN SCHEMA RETAIL_ANALYTICS.SALES;
