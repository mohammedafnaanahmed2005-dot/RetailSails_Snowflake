-- 03_load_csv.sql
-- Loads the sample CSV through the internal stage with COPY INTO.
-- Step 1 (client side, from SnowSQL / snow CLI / scripts/load_sample.py):
--   PUT file://data/retail_sales_analytics_sample.csv @RETAIL_ANALYTICS.SALES.RAW_STAGE/sample/ AUTO_COMPRESS=TRUE OVERWRITE=TRUE;
USE ROLE RETAIL_ANALYST;
USE WAREHOUSE RETAIL_WH;
USE SCHEMA RETAIL_ANALYTICS.SALES;

-- Step 2: dry run - scan the staged file for parse problems without loading anything.
-- (VALIDATION_MODE cannot be combined with a column list, so TRY_* casts are used instead.)
SELECT COUNT(*) AS rows_in_file,
       COUNT_IF(TRY_TO_DATE($2) IS NULL)          AS bad_dates,
       COUNT_IF(TRY_TO_NUMBER($8) IS NULL)        AS bad_qty,
       COUNT_IF(TRY_TO_DECIMAL($9,14,2) IS NULL OR TRY_TO_DECIMAL($20,16,2) IS NULL) AS bad_money,
       COUNT_IF(TRY_TO_BOOLEAN($17) IS NULL)      AS bad_bool,
       COUNT_IF($1 IS NULL)                       AS null_ids,
       COUNT(*) - COUNT(DISTINCT $1)              AS dup_ids
FROM @RAW_STAGE/sample/ (FILE_FORMAT => 'CSV_FF');

-- Step 3: real load. ON_ERROR = ABORT_STATEMENT => all-or-nothing, never silently drops rows.
-- Only load if the sample has not been loaded already (idempotent rerun).
EXECUTE IMMEDIATE $$
BEGIN
  LET n NUMBER := (SELECT COUNT(*) FROM SALES_TRANSACTIONS WHERE SOURCE_FILE LIKE '%retail_sales_analytics_sample.csv%');
  IF (n = 0) THEN
    COPY INTO SALES_TRANSACTIONS (
      TRANSACTION_ID, TRANSACTION_DATE, PRODUCT_NAME, CATEGORY, REGION, SALES_CHANNEL,
      CUSTOMER_SEGMENT, QUANTITY, UNIT_PRICE_INR, DISCOUNT_PCT, GROSS_REVENUE_INR,
      DISCOUNT_AMOUNT_INR, REVENUE_INR, COST_OF_GOODS_INR, SHIPPING_COST_INR,
      MARKETING_COST_INR, RETURNED, RETURN_AMOUNT_INR, NET_REVENUE_INR, NET_PROFIT_INR,
      PAYMENT_METHOD, PROMOTION_TYPE, SOURCE_FILE, UPLOAD_ID)
    FROM (SELECT $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,
                 METADATA$FILENAME, 'SAMPLE_INITIAL_LOAD'
          FROM @RAW_STAGE/sample/)
    FILE_FORMAT = (FORMAT_NAME = CSV_FF)
    ON_ERROR = ABORT_STATEMENT;

    INSERT INTO UPLOAD_HISTORY (UPLOAD_ID, FILE_NAME, TARGET_TABLE, LOAD_MODE, ROWS_IN_FILE, ROWS_LOADED, ROWS_REJECTED, STATUS, VALIDATION_SUMMARY)
      SELECT 'SAMPLE_INITIAL_LOAD', 'retail_sales_analytics_sample.csv', 'SALES_TRANSACTIONS', 'COPY_INTO',
             COUNT(*), COUNT(*), 0, 'SUCCESS', OBJECT_CONSTRUCT('method','PUT + COPY INTO', 'on_error','ABORT_STATEMENT')
      FROM SALES_TRANSACTIONS WHERE UPLOAD_ID = 'SAMPLE_INITIAL_LOAD';
    RETURN 'loaded';
  END IF;
  RETURN 'already loaded - skipped';
END;
$$;

SELECT COUNT(*) AS ROW_COUNT, MIN(TRANSACTION_DATE), MAX(TRANSACTION_DATE) FROM SALES_TRANSACTIONS;
