"""Load application-time Gold features with labels mature by a chosen date."""

from datetime import date
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


CATEGORICAL_FEATURES = (
    "Occupation",
    "Credit_Mix",
    "Payment_of_Min_Amount",
    "Payment_Behaviour",
)

NUMERIC_FEATURES = (
    "tenure",
    "loan_amt",
    "Age",
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Num_Credit_Inquiries",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Monthly_Balance",
    "Amount_invested_monthly_special_code",
    "Type_of_Loan_has_unknown",
    "Credit_History_Months",
    "debt_to_salary",
    "emi_to_salary",
    "repayment_ability",
    "num_fin_products",
    "click_history_months",
    "click_history_missing",
    *(f"fe_{i}_mean_3m" for i in range(1, 21)),
    *(f"fe_{i}_latest" for i in range(1, 21)),
)

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_modelling_dataset(
    spark: SparkSession, gold_directory: str | Path, as_of_date: str
) -> DataFrame:
    """Join Gold by loan_id, retaining only labels observable by as_of_date.

    Pass the model fitting date for Train/Validation and the later evaluation
    date for OOT. No missing-value imputation or feature fitting happens here.
    """
    cutoff = date.fromisoformat(as_of_date)
    gold_directory = Path(gold_directory)
    features = spark.read.parquet(str(gold_directory / "feature_store"))
    labels = spark.read.parquet(str(gold_directory / "label_store"))
    required = {"loan_id", "Customer_ID", "feature_snapshot_date", *MODEL_FEATURES}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"feature_store is missing model columns: {sorted(missing)}")

    return (
        features.select("loan_id", "Customer_ID", "feature_snapshot_date", *MODEL_FEATURES)
        .join(labels.select("loan_id", "label_snapshot_date", "label"), "loan_id", "inner")
        .filter(F.col("label_snapshot_date") <= F.lit(cutoff))
        .select(
            "loan_id",
            "Customer_ID",
            "feature_snapshot_date",
            "label_snapshot_date",
            "label",
            *MODEL_FEATURES,
        )
    )
