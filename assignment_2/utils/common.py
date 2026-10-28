from functools import reduce
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType


SOURCE_FILES = {
    "attributes": "features_attributes.csv",
    "financials": "features_financials.csv",
    "clickstream": "feature_clickstream.csv",
    "lms": "lms_loan_daily.csv",
}

SOURCE_COLUMNS = {
    "attributes": ["Customer_ID", "Name", "Age", "SSN", "Occupation", "snapshot_date"],
    "financials": [
        "Customer_ID", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
        "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Type_of_Loan",
        "Delay_from_due_date", "Num_of_Delayed_Payment", "Changed_Credit_Limit",
        "Num_Credit_Inquiries", "Credit_Mix", "Outstanding_Debt",
        "Credit_Utilization_Ratio", "Credit_History_Age", "Payment_of_Min_Amount",
        "Total_EMI_per_month", "Amount_invested_monthly", "Payment_Behaviour",
        "Monthly_Balance", "snapshot_date",
    ],
    "clickstream": [f"fe_{i}" for i in range(1, 21)] + ["Customer_ID", "snapshot_date"],
    "lms": [
        "loan_id", "Customer_ID", "loan_start_date", "tenure", "installment_num",
        "loan_amt", "due_amt", "paid_amt", "overdue_amt", "balance", "snapshot_date",
    ],
}


def read_csv_strings(path, source, spark):
    """Read source or Bronze CSVs without interpreting numeric or missing codes."""
    schema = StructType([StructField(name, StringType()) for name in SOURCE_COLUMNS[source]])
    return (
        spark.read.schema(schema)
        .option("header", True)
        .option("enforceSchema", False)
        .option("mode", "FAILFAST")
        .option("escape", '"')
        .option("ignoreLeadingWhiteSpace", False)
        .option("ignoreTrailingWhiteSpace", False)
        .option("nullValue", "\u0000")
        .option("emptyValue", "")
        .csv(str(path))
        .fillna("")  # Represent empty CSV fields as empty strings.
    )


def validate_keys(df, source):
    """Fail on missing identifiers/dates or duplicate keys; never silently drop rows."""
    keys = ["loan_id", "snapshot_date"] if source == "lms" else ["Customer_ID", "snapshot_date"]
    required = list(dict.fromkeys(keys + ["Customer_ID"]))
    missing = [F.col(c).isNull() | (F.trim(F.col(c).cast("string")) == "") for c in required]
    if df.filter(reduce(lambda a, b: a | b, missing)).limit(1).count():
        raise ValueError(f"{source}: missing identifier or snapshot date")
    if df.filter(F.to_date("snapshot_date", "yyyy-MM-dd").isNull()).limit(1).count():
        raise ValueError(f"{source}: invalid snapshot date")
    if df.groupBy(*keys).count().filter(F.col("count") > 1).limit(1).count():
        raise ValueError(f"{source}: duplicate key {keys}")
