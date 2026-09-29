"""Build application-time feature and outcome label stores from Silver tables."""

from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


CLICKSTREAM_COLUMNS = [f"fe_{i}" for i in range(1, 21)]
CLICKSTREAM_LOOKBACK_MONTHS = 3
LABEL_DPD = 30
LABEL_MOB = 6


def build_clickstream_features(
    applications: DataFrame,
    clickstream: DataFrame,
    lookback_months: int = CLICKSTREAM_LOOKBACK_MONTHS,
) -> DataFrame:
    """Aggregate only clickstream observations known before each application."""
    app = applications.alias("app")
    clicks = clickstream.alias("click")
    history = app.join(
        clicks,
        (F.col("app.Customer_ID") == F.col("click.Customer_ID"))
        & (F.col("click.snapshot_date") >= F.add_months(F.col("app.feature_snapshot_date"), -lookback_months))
        & (F.col("click.snapshot_date") < F.col("app.feature_snapshot_date")),
        "left",
    ).select(
        F.col("app.loan_id"),
        F.col("click.snapshot_date").alias("click_snapshot_date"),
        *[F.col(f"click.{column}").alias(column) for column in CLICKSTREAM_COLUMNS],
    )

    aggregations = [
        F.count("click_snapshot_date").cast("int").alias("click_history_months"),
        *[F.avg(column).alias(f"{column}_mean_3m") for column in CLICKSTREAM_COLUMNS],
        *[
            F.max_by(F.col(column), F.col("click_snapshot_date")).alias(f"{column}_latest")
            for column in CLICKSTREAM_COLUMNS
        ],
    ]
    return (
        history.groupBy("loan_id")
        .agg(*aggregations)
        .withColumn(
            "click_history_missing",
            (F.col("click_history_months") == 0).cast("int"),
        )
    )


def build_feature_store(
    attributes: DataFrame,
    financials: DataFrame,
    clickstream: DataFrame,
    loan_daily: DataFrame,
) -> DataFrame:
    """Create one leakage-safe feature row for every loan application."""
    applications = (
        loan_daily.filter(F.col("mob") == 0)
        .select(
            "loan_id",
            "Customer_ID",
            F.col("loan_start_date").alias("feature_snapshot_date"),
            "tenure",
            "loan_amt",
        )
    )
    attribute_features = attributes.withColumnRenamed("snapshot_date", "feature_snapshot_date")
    financial_features = financials.withColumnRenamed("snapshot_date", "feature_snapshot_date")
    # __10000__ remains traceable through its flag but is not treated as a genuine investment amount.
    financial_features = financial_features.withColumn(
        "Amount_invested_monthly",
        F.when(
            F.col("Amount_invested_monthly_special_code") == 0,
            F.col("Amount_invested_monthly"),
        ),
    )
    click_features = build_clickstream_features(applications, clickstream)

    return (
        applications.join(
            attribute_features,
            ["Customer_ID", "feature_snapshot_date"],
            "left",
        )
        .join(
            financial_features,
            ["Customer_ID", "feature_snapshot_date"],
            "left",
        )
        .join(click_features, "loan_id", "left")
        .select(
            "loan_id",
            "Customer_ID",
            "feature_snapshot_date",
            *[
                column
                for column in (
                    applications.columns[2:]
                    + [c for c in attribute_features.columns if c not in {"Customer_ID", "feature_snapshot_date"}]
                    + [c for c in financial_features.columns if c not in {"Customer_ID", "feature_snapshot_date"}]
                    + [c for c in click_features.columns if c != "loan_id"]
                )
                if column != "feature_snapshot_date"
            ],
        )
    )


def build_label_store(
    loan_daily: DataFrame,
    dpd_threshold: int = LABEL_DPD,
    observation_mob: int = LABEL_MOB,
) -> DataFrame:
    """Create the Lab 2 definition: default when DPD is at least 30 at MOB 6."""
    return (
        loan_daily.filter(F.col("mob") == observation_mob)
        .withColumn(
            "label",
            F.when(F.col("dpd") >= dpd_threshold, 1).otherwise(0).cast("int"),
        )
        .withColumn("label_def", F.lit(f"{dpd_threshold}dpd_{observation_mob}mob"))
        .select(
            "loan_id",
            "Customer_ID",
            "label",
            "label_def",
            F.col("snapshot_date").alias("label_snapshot_date"),
        )
    )


def _validate_unique(df: DataFrame, key: str, table_name: str) -> None:
    if df.filter(F.col(key).isNull()).limit(1).count():
        raise ValueError(f"{table_name}: missing {key}")
    if df.groupBy(key).count().filter(F.col("count") > 1).limit(1).count():
        raise ValueError(f"{table_name}: duplicate {key}")


def process_gold_tables(silver_directory, gold_directory, spark):
    """Read Silver tables, build both Gold stores, validate them, and write Parquet."""
    silver_directory = Path(silver_directory)
    gold_directory = Path(gold_directory)
    attributes = spark.read.parquet(str(silver_directory / "attributes"))
    financials = spark.read.parquet(str(silver_directory / "financials"))
    clickstream = spark.read.parquet(str(silver_directory / "clickstream"))
    loan_daily = spark.read.parquet(str(silver_directory / "loan_daily"))

    feature_store = build_feature_store(attributes, financials, clickstream, loan_daily).cache()
    label_store = build_label_store(loan_daily).cache()
    try:
        _validate_unique(feature_store, "loan_id", "feature_store")
        _validate_unique(label_store, "loan_id", "label_store")
        applications = (
            loan_daily.filter(F.col("mob") == 0)
            .select("Customer_ID", F.col("loan_start_date").alias("feature_snapshot_date"))
        )
        application_count = applications.count()
        if feature_store.count() != application_count:
            raise ValueError("feature_store: not every application produced exactly one row")
        if label_store.count() != application_count:
            raise ValueError("label_store: not every application has a MOB 6 label")
        for name, source in (("attributes", attributes), ("financials", financials)):
            source_keys = source.select(
                "Customer_ID",
                F.col("snapshot_date").alias("feature_snapshot_date"),
            )
            if applications.join(
                source_keys,
                ["Customer_ID", "feature_snapshot_date"],
                "left_anti",
            ).limit(1).count():
                raise ValueError(f"feature_store: application is missing its {name} row")

        feature_path = gold_directory / "feature_store"
        label_path = gold_directory / "label_store"
        (
            feature_store.repartition("feature_snapshot_date")
            .write.mode("overwrite")
            .partitionBy("feature_snapshot_date")
            .parquet(str(feature_path))
        )
        (
            label_store.repartition("label_snapshot_date")
            .write.mode("overwrite")
            .partitionBy("label_snapshot_date")
            .parquet(str(label_path))
        )
        print(f"Gold feature_store: {application_count:,} rows -> {feature_path}")
        print(f"Gold label_store: {application_count:,} rows -> {label_path}")
        return feature_store, label_store
    finally:
        feature_store.unpersist()
        label_store.unpersist()
