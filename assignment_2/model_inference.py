"""Score one application month with the frozen champion; no training occurs here."""

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

from utils.model_store import load_model


ROOT = Path(__file__).resolve().parent
MODEL_VERSION = "xgboost_20241031_v1"
PREDICTION_COLUMNS = {
    "loan_id",
    "Customer_ID",
    "feature_snapshot_date",
    "model_version",
    "predicted_default_probability",
}
OUTCOME_COLUMNS = {"label", "label_snapshot_date", "dpd", "overdue_amt", "paid_amt"}


def score_month(snapshot_date: str) -> dict:
    snapshot = date.fromisoformat(snapshot_date)
    if snapshot.day != 1:
        raise ValueError("snapshot-date must be the first day of an application month")

    pipeline, metadata = load_model(ROOT / "model_bank", MODEL_VERSION)
    if snapshot <= date.fromisoformat(metadata["fit_date"]):
        raise ValueError("Inference month must be after the model fitting cut-off")
    input_features = metadata["input_features"]
    if len(input_features) != metadata["input_feature_count"] or len(set(input_features)) != len(input_features):
        raise ValueError("Model metadata has an invalid input feature list")
    if input_features != metadata["numeric_features"] + metadata["categorical_features"]:
        raise ValueError("Model metadata feature groups do not match input order")
    if set(input_features) & (OUTCOME_COLUMNS | {"loan_id", "Customer_ID", "feature_snapshot_date"}):
        raise ValueError("Model input list contains an ID, date, or outcome field")
    if list(pipeline.named_steps["preprocessing"].feature_names_in_) != input_features:
        raise ValueError("Saved pipeline input order does not match model metadata")

    spark = (
        SparkSession.builder.appName("assignment2-monthly-inference")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    try:
        feature_store = spark.read.parquet(str(ROOT / "datamart" / "gold" / "feature_store"))
        missing = set(input_features) - set(feature_store.columns)
        if missing:
            raise ValueError(f"feature_store is missing required fields: {sorted(missing)}")
        source_types = dict(feature_store.dtypes)
        invalid_numeric = [
            name for name in metadata["numeric_features"]
            if source_types[name] not in {"int", "bigint", "float", "double"}
        ]
        invalid_categorical = [
            name for name in metadata["categorical_features"]
            if source_types[name] != "string"
        ]
        if invalid_numeric or invalid_categorical:
            raise ValueError(
                f"feature_store has incompatible types: {invalid_numeric + invalid_categorical}"
            )
        frame = (
            feature_store.filter(F.col("feature_snapshot_date") == F.lit(snapshot))
            .select("loan_id", "Customer_ID", "feature_snapshot_date", *input_features)
            .toPandas()
        )
        if frame.empty:
            raise ValueError(f"No applications found for {snapshot_date}")
        if frame["loan_id"].isna().any() or not frame["loan_id"].is_unique:
            raise ValueError("Monthly feature rows need unique, non-null loan_id")
        if frame["Customer_ID"].isna().any():
            raise ValueError("Monthly feature rows contain null Customer_ID")
        if any(str(value) != snapshot_date for value in frame["feature_snapshot_date"]):
            raise ValueError("Feature snapshot date differs from requested month")

        probabilities = np.asarray(
            pipeline.predict_proba(frame.loc[:, input_features])[:, 1], dtype=float
        )
        if len(probabilities) != len(frame) or not np.isfinite(probabilities).all():
            raise ValueError("Prediction count or probability values are invalid")
        if ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError("Predicted probability falls outside [0, 1]")

        records = [
            (str(loan_id), str(customer_id), metadata["model_version"], float(probability))
            for loan_id, customer_id, probability in zip(
                frame["loan_id"], frame["Customer_ID"], probabilities
            )
        ]
        schema = StructType(
            [
                StructField("loan_id", StringType(), nullable=False),
                StructField("Customer_ID", StringType(), nullable=False),
                StructField("model_version", StringType(), nullable=False),
                StructField("predicted_default_probability", DoubleType(), nullable=False),
            ]
        )
        prediction_root = ROOT / "datamart" / "gold" / "model_predictions"
        month_path = prediction_root / f"feature_snapshot_date={snapshot_date}"
        # Overwrite only the requested month; other partition folders remain intact.
        spark.createDataFrame(records, schema).coalesce(1).write.mode("overwrite").parquet(
            str(month_path)
        )

        stored = (
            spark.read.parquet(str(prediction_root))
            .filter(F.col("feature_snapshot_date") == F.lit(snapshot))
            .toPandas()
        )
        if set(stored.columns) != PREDICTION_COLUMNS:
            raise ValueError("Stored prediction schema is not the expected Gold schema")
        if len(stored) != len(frame) or set(stored["loan_id"]) != set(frame["loan_id"]):
            raise ValueError("Stored predictions do not cover the eligible applications")
        if stored["loan_id"].isna().any() or stored.duplicated(["loan_id", "model_version"]).any():
            raise ValueError("Stored predictions contain null or duplicate loan IDs")
        if any(str(value) != snapshot_date for value in stored["feature_snapshot_date"]):
            raise ValueError("Stored predictions have an unexpected application month")
        if not stored["model_version"].eq(MODEL_VERSION).all():
            raise ValueError("Stored predictions have an unexpected model version")
        scores = stored["predicted_default_probability"]
        if scores.isna().any() or not scores.between(0, 1).all():
            raise ValueError("Stored prediction probabilities are invalid")

        return {
            "snapshot_date": snapshot_date,
            "model_version": MODEL_VERSION,
            "rows": len(stored),
            "unique_loan_ids": stored["loan_id"].nunique(),
            "minimum_probability": float(scores.min()),
            "maximum_probability": float(scores.max()),
            "mean_probability": float(scores.mean()),
            "median_probability": float(scores.median()),
            "partition_path": str(month_path),
        }
    finally:
        spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score one loan application month")
    parser.add_argument("--snapshot-date", required=True, help="First day of month, YYYY-MM-01")
    print("INFERENCE_RESULT=" + json.dumps(score_month(parser.parse_args().snapshot_date)))
