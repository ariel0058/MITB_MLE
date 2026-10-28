"""Create three static charts from the monthly monitoring Gold table."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql import SparkSession


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "reports" / "monitoring"


def main() -> None:
    spark = (
        SparkSession.builder.appName("assignment2-monitoring-charts")
        .master("local[2]")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    try:
        data = spark.read.parquet(str(ROOT / "datamart" / "gold" / "model_monitoring")).toPandas()
    finally:
        spark.stop()
    data = data.sort_values("feature_snapshot_date")
    if len(data) != 3 or data["model_version"].nunique() != 1:
        raise ValueError("Expected exactly three monitoring months for one model version")
    months = pd.to_datetime(data["feature_snapshot_date"]).dt.strftime("%b %Y")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(months, data["roc_auc"], marker="o", linewidth=2, label="ROC-AUC")
    ax.plot(months, data["gini"], marker="s", linewidth=2, label="Gini")
    ax.set(title="Model discrimination by application month", ylabel="Metric", ylim=(0, 1))
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "performance.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(months, 100 * data["mean_predicted_probability"], marker="o", linewidth=2, label="Mean predicted risk")
    ax.plot(months, 100 * data["actual_default_rate"], marker="s", linewidth=2, label="Actual default rate")
    ax.set(title="Predicted risk and observed default rate", ylabel="Percent (%)", ylim=(0, 40))
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "risk_vs_actual.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(months, data["score_psi"], color="#3973a8")
    ax.axhline(0.10, color="#d29b2d", linestyle="--", linewidth=1, label="0.10 guideline")
    ax.axhline(0.25, color="#b94a48", linestyle="--", linewidth=1, label="0.25 guideline")
    ax.set(title="Score distribution stability vs OOT reference", ylabel="Score PSI", ylim=(0, 0.30))
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "score_psi.png", dpi=170)
    plt.close(fig)

    print("MONITORING_PLOTS=" + ", ".join(str(path) for path in sorted(OUTPUT.glob("*.png"))))


if __name__ == "__main__":
    main()
