"""Clean Bronze sources and write typed Silver Parquet tables."""

from pathlib import Path
from pyspark.sql import functions as F
from utils.common import read_csv_strings, validate_keys


AGE_MIN, AGE_MAX = 0, 150
MISSING_TEXT = ["", "nan", "null", "none", "n/a", "na"]
BALANCE_SENTINEL = "-333333333333333333333333333" 
PAYMENT_BEHAVIOURS = [
    "Low_spent_Small_value_payments", "Low_spent_Medium_value_payments",
    "Low_spent_Large_value_payments", "High_spent_Small_value_payments",
    "High_spent_Medium_value_payments", "High_spent_Large_value_payments",
]


def numeric_value(column, *, integer=False, minimum=None, maximum=None, invalid_values=()):
    """Strip edge underscores and map invalid numbers to null, without imputation."""
    text = F.regexp_replace(F.trim(F.col(column)), r"^_+|_+$", "")
    number_format = r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"
    number = F.when(text.rlike(number_format), text.cast("double"))
    valid = number.isNotNull() & ~F.isnan(number) & (F.abs(number) != float("inf"))
    if integer:
        valid = valid & (number % 1 == 0) & number.between(-2147483648, 2147483647)
    if minimum is not None:
        valid = valid & (number >= minimum)
    if maximum is not None:
        valid = valid & (number <= maximum)
    if invalid_values:
        valid = valid & ~text.isin(*invalid_values)
    return F.when(valid, number).cast("int" if integer else "double")


def category_value(column):
    text = F.trim(F.col(column))
    missing = text.isNull() | F.lower(text).isin(*MISSING_TEXT) | text.rlike(r"^_+$")
    return F.when(missing, "Unknown").otherwise(text)


def clean_attributes(df):
    return df.select(
        "Customer_ID", "snapshot_date",
        numeric_value("Age", integer=True, minimum=AGE_MIN, maximum=AGE_MAX).alias("Age"),
        category_value("Occupation").alias("Occupation"),
    )  # Name and SSN remain available in Bronze, not in the feature tables.


def clean_financials(df):
    nonnegative_amounts = [
        "Annual_Income", "Monthly_Inhand_Salary", "Interest_Rate", "Outstanding_Debt",
        "Total_EMI_per_month", "Amount_invested_monthly",
    ]
    count_columns = [
        "Num_Bank_Accounts", "Num_Credit_Card", "Num_of_Loan",
        "Num_of_Delayed_Payment", "Num_Credit_Inquiries",
    ]
    # Keep this unresolved code visible rather than assuming it is a valid amount.
    df = df.withColumn(
        "Amount_invested_monthly_special_code",
        (F.trim(F.col("Amount_invested_monthly")) == "__10000__").cast("int"),
    )
    conversions = {c: numeric_value(c, minimum=0) for c in nonnegative_amounts}
    conversions.update({c: numeric_value(c, integer=True, minimum=0) for c in count_columns})
    conversions.update({
        "Delay_from_due_date": numeric_value("Delay_from_due_date", integer=True),
        "Changed_Credit_Limit": numeric_value("Changed_Credit_Limit"),
        "Credit_Utilization_Ratio": numeric_value("Credit_Utilization_Ratio", minimum=0, maximum=100),
        "Monthly_Balance": numeric_value("Monthly_Balance", invalid_values=(BALANCE_SENTINEL,)),
        "Credit_Mix": category_value("Credit_Mix"),
        "Payment_of_Min_Amount": category_value("Payment_of_Min_Amount"),
    })
    df = df.withColumns(conversions)
    df = df.withColumn(
        "Credit_Mix",
        F.when(F.col("Credit_Mix").isin("Standard", "Good", "Bad"), F.col("Credit_Mix"))
        .otherwise("Unknown"),
    )
    behaviour = F.trim(F.col("Payment_Behaviour"))
    df = df.withColumn(
        "Payment_Behaviour",
        F.when(behaviour.isin(*PAYMENT_BEHAVIOURS), behaviour).otherwise("Unknown"),
    )

    raw_type = F.trim(F.col("Type_of_Loan"))
    loan_type = F.regexp_replace(category_value("Type_of_Loan"), r"(?i)\bNot Specified\b", "Unknown")
    df = df.withColumn(
        "Type_of_Loan",
        F.when((F.col("Num_of_Loan") == 0) & (raw_type == ""), "No_Loan").otherwise(loan_type),
    )
    df = df.withColumn("Type_of_Loan_has_unknown", F.col("Type_of_Loan").rlike(r"\bUnknown\b").cast("int"))
    # Preserve known types in mixed lists; NM remains a separate category.
    history = F.trim(F.col("Credit_History_Age"))
    pattern = r"^(\d+) Years? and (\d+) Months?$"
    years = F.when(history.rlike(pattern), F.regexp_extract(history, pattern, 1).cast("int"))
    months = F.when(history.rlike(pattern), F.regexp_extract(history, pattern, 2).cast("int"))
    df = df.withColumn(
        "Credit_History_Months",
        F.when(months.between(0, 11), years * 12 + months).cast("int"),
    )
    return df.drop("Credit_History_Age")


def clean_clickstream(df):
    # Keep all snapshots and signed values. Gold will apply application-time filters.
    return df.withColumns({f"fe_{i}": numeric_value(f"fe_{i}") for i in range(1, 21)})


def clean_loans(df):
    conversions = {
        c: numeric_value(c, minimum=0)
        for c in ["loan_amt", "due_amt", "paid_amt", "overdue_amt", "balance"]
    }
    conversions.update({
        "loan_id": F.trim(F.col("loan_id")),
        "loan_start_date": F.to_date("loan_start_date", "yyyy-MM-dd"),
        "tenure": numeric_value("tenure", integer=True, minimum=1),
        "installment_num": numeric_value("installment_num", integer=True, minimum=0),
    })
    df = df.withColumns(conversions).withColumn("mob", F.col("installment_num"))
    due, overdue = F.col("due_amt"), F.col("overdue_amt")
    missed = F.when(due.isNotNull() & overdue.isNotNull(),
        F.when(overdue == 0, 0).when(due > 0, F.ceil(overdue / due))
    )
    df = df.withColumn("installments_missed", missed.cast("int"))
    df = df.withColumn(
        "first_missed_date",
        F.when(F.col("installments_missed") > 0,
               F.add_months("snapshot_date", -F.col("installments_missed"))),
    )
    return df.withColumn(
        "dpd",
        F.when(F.col("installments_missed") == 0, 0)
        .when(F.col("installments_missed") > 0, F.datediff("snapshot_date", "first_missed_date"))
        .cast("int"),
    )


TRANSFORMS = {
    "attributes": clean_attributes,
    "financials": clean_financials,
    "clickstream": clean_clickstream,
    "lms": clean_loans,
}


def process_silver_table(source, bronze_directory, silver_directory, spark):
    """Read Bronze only, save all dated snapshots."""
    input_path = Path(bronze_directory) / source
    folder = "loan_daily" if source == "lms" else source
    output_path = Path(silver_directory) / folder
    df = read_csv_strings(input_path, source, spark)
    df = df.withColumns({
        "Customer_ID": F.trim(F.col("Customer_ID")),
        "snapshot_date": F.to_date("snapshot_date", "yyyy-MM-dd"),
    })
    df = TRANSFORMS[source](df).cache()
    try:
        validate_keys(df, source)
        rows = df.count()
        dates = df.select("snapshot_date").distinct().count()
        (
            df.repartition("snapshot_date").write.mode("overwrite")
            .partitionBy("snapshot_date").parquet(str(output_path))
        )
        print(f"Silver {source}: {rows:,} rows, {dates} dates -> {output_path}")
        return df
    finally:
        df.unpersist()
