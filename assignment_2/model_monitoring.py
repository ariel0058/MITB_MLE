"""Calculate score stability and mature-label performance for one month."""

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
from dateutil.relativedelta import relativedelta
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DateType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)
from sklearn.metrics import roc_auc_score

from utils.model_store import load_model


ROOT = Path(__file__).resolve().parent
MODEL_VERSION = "xgboost_20241031_v1"
EPSILON = 1e-6


def score_psi(reference_scores, current_scores) -> tuple[float, int]:
    """Use reference-only decile boundaries for both score populations."""
    internal_edges = np.unique(np.quantile(reference_scores, np.arange(0.1, 1.0, 0.1)))
    edges = np.r_[-np.inf, internal_edges, np.inf]
    reference_counts, _ = np.histogram(reference_scores, bins=edges)
    current_counts, _ = np.histogram(current_scores, bins=edges)
    reference_share = reference_counts / len(reference_scores)
    current_share = current_counts / len(current_scores)
    safe_reference = np.maximum(reference_share, EPSILON)
    safe_current = np.maximum(current_share, EPSILON)
    psi = np.sum((safe_current - safe_reference) * np.log(safe_current / safe_reference))
    return max(0.0, float(psi)), len(edges) - 1


def monitor_month(snapshot_date: str, as_of_date: str, dry_run: bool = False) -> dict:
    snapshot = date.fromisoformat(snapshot_date)
    as_of = date.fromisoformat(as_of_date)
    if snapshot.day != 1:
        raise ValueError("snapshot-date must be the first day of an application month")
    if as_of < snapshot:
        raise ValueError("as-of date cannot be earlier than the prediction/application month")

    pipeline, metadata = load_model(ROOT / "model_bank", MODEL_VERSION)
    if metadata["input_feature_count"] != len(metadata["input_features"]):
        raise ValueError("Model metadata feature count is inconsistent")
    if list(pipeline.named_steps["preprocessing"].feature_names_in_) != metadata["input_features"]:
        raise ValueError("Saved model input order differs from metadata")

    spark = (
        SparkSession.builder.appName("assignment2-monthly-monitoring")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    try:
        predictions = (
            spark.read.parquet(str(ROOT / "datamart" / "gold" / "model_predictions"))
            .filter(F.col("feature_snapshot_date") == F.lit(snapshot))
            .toPandas()
        )
        if predictions.empty:
            raise ValueError(f"No stored predictions found for {snapshot_date}")
        if predictions["loan_id"].isna().any() or not predictions["loan_id"].is_unique:
            raise ValueError("Monthly predictions have null or duplicate loan IDs")
        if not predictions["model_version"].eq(MODEL_VERSION).all():
            raise ValueError("Monthly predictions contain another model version")
        current_scores = predictions["predicted_default_probability"].to_numpy(dtype=float)
        if not np.isfinite(current_scores).all() or ((current_scores < 0) | (current_scores > 1)).any():
            raise ValueError("Monthly predictions contain invalid probabilities")

        features = spark.read.parquet(str(ROOT / "datamart" / "gold" / "feature_store"))
        missing = set(metadata["input_features"]) - set(features.columns)
        if missing:
            raise ValueError(f"OOT feature store is missing model inputs: {sorted(missing)}")
        reference = (
            features.filter(
                (F.col("feature_snapshot_date") >= "2024-05-01")
                & (F.col("feature_snapshot_date") < "2024-11-01")
            )
            .select("loan_id", *metadata["input_features"])
            .toPandas()
        )
        if len(reference) != 2986 or not reference["loan_id"].is_unique:
            raise ValueError("Expected 2,986 unique OOT reference applications")
        reference_scores = pipeline.predict_proba(reference.loc[:, metadata["input_features"]])[:, 1]
        if not np.isfinite(reference_scores).all():
            raise ValueError("OOT reference scores contain invalid values")
        psi, bin_count = score_psi(reference_scores, current_scores)

        expected_label_date = snapshot + relativedelta(months=6)
        performance_available = as_of >= expected_label_date
        labelled_count = 0
        actual_default_rate = None
        auc = None
        gini = None
        if performance_available:
            labels = (
                spark.read.parquet(str(ROOT / "datamart" / "gold" / "label_store"))
                .filter(F.col("label_snapshot_date") == F.lit(expected_label_date))
                .select("loan_id", "Customer_ID", "label", "label_def", "label_snapshot_date")
                .toPandas()
            )
            if labels["loan_id"].isna().any() or not labels["loan_id"].is_unique:
                raise ValueError("Mature label rows contain null or duplicate loan IDs")
            joined = predictions.merge(labels, on="loan_id", how="left", validate="one_to_one", indicator=True)
            if len(joined) != len(predictions) or not joined["_merge"].eq("both").all():
                raise ValueError("Every matured prediction must match one label")
            if not joined["Customer_ID_x"].eq(joined["Customer_ID_y"]).all():
                raise ValueError("Prediction and label customer IDs differ")
            if not joined["label"].isin([0, 1]).all():
                raise ValueError("Mature labels must be binary")
            if not joined["label_def"].eq(metadata["target_definition"]).all():
                raise ValueError("Mature label definition differs from the model target")
            if not joined["label_snapshot_date"].eq(expected_label_date).all():
                raise ValueError("Mature label date does not equal application date + 6 months")
            labelled_count = len(joined)
            actual_default_rate = float(joined["label"].mean())
            if joined["label"].nunique() != 2:
                raise ValueError("ROC-AUC requires both label classes")
            auc = float(roc_auc_score(joined["label"], joined["predicted_default_probability"]))
            gini = 2 * auc - 1

        result = {
            "feature_snapshot_date": snapshot_date,
            "model_version": MODEL_VERSION,
            "prediction_count": len(predictions),
            "mean_predicted_probability": float(current_scores.mean()),
            "median_predicted_probability": float(np.median(current_scores)),
            "score_psi": psi,
            "reference_count": len(reference),
            "reference_bin_count": bin_count,
            "expected_label_snapshot_date": expected_label_date.isoformat(),
            "performance_as_of_date": as_of_date,
            "performance_available": performance_available,
            "labelled_count": labelled_count,
            "actual_default_rate": actual_default_rate,
            "roc_auc": auc,
            "gini": gini,
            "dry_run": dry_run,
        }
        if not dry_run:
            schema = StructType(
                [
                    StructField("model_version", StringType(), nullable=False),
                    StructField("prediction_count", IntegerType(), nullable=False),
                    StructField("mean_predicted_probability", DoubleType(), nullable=False),
                    StructField("median_predicted_probability", DoubleType(), nullable=False),
                    StructField("score_psi", DoubleType(), nullable=False),
                    StructField("expected_label_snapshot_date", DateType(), nullable=False),
                    StructField("performance_as_of_date", DateType(), nullable=False),
                    StructField("performance_available", BooleanType(), nullable=False),
                    StructField("labelled_count", IntegerType(), nullable=False),
                    StructField("actual_default_rate", DoubleType(), nullable=True),
                    StructField("roc_auc", DoubleType(), nullable=True),
                    StructField("gini", DoubleType(), nullable=True),
                ]
            )
            row = (
                MODEL_VERSION,
                len(predictions),
                result["mean_predicted_probability"],
                result["median_predicted_probability"],
                psi,
                expected_label_date,
                as_of,
                performance_available,
                labelled_count,
                actual_default_rate,
                auc,
                gini,
            )
            monitoring_root = ROOT / "datamart" / "gold" / "model_monitoring"
            month_path = monitoring_root / f"feature_snapshot_date={snapshot_date}"
            spark.createDataFrame([row], schema).coalesce(1).write.mode("overwrite").parquet(
                str(month_path)
            )
            stored = spark.read.parquet(str(monitoring_root)).filter(
                F.col("feature_snapshot_date") == F.lit(snapshot)
            )
            if stored.count() != 1:
                raise ValueError("Monitoring partition must contain exactly one row")
            result["partition_path"] = str(month_path)
        return result
    finally:
        spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor one prediction month")
    parser.add_argument("--snapshot-date", required=True, help="Application month, YYYY-MM-01")
    parser.add_argument("--as-of-date", required=True, help="Simulated knowledge date, YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true", help="Calculate without writing Gold")
    arguments = parser.parse_args()
    print(
        "MONITORING_RESULT="
        + json.dumps(monitor_month(arguments.snapshot_date, arguments.as_of_date, arguments.dry_run))
    )
