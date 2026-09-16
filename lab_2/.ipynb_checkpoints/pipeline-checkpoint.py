import argparse
import os

import pyspark


from utils.data_processing_bronze_table import process_bronze_table
from utils.data_processing_silver_table import process_silver_table
from utils.data_processing_gold_table import process_labels_gold_table


def main(snapshotdate):
    print("\n\n--- starting incremental pipeline ---\n")
    print(f"snapshot date: {snapshotdate}\n")

    # 1. Initialize SparkSession
    spark = (
        pyspark.sql.SparkSession.builder
        .appName("incremental_pipeline")
        .master("local[*]")
        .getOrCreate()
    )

    # Hide Spark warning messages
    spark.sparkContext.setLogLevel("ERROR")

    try:
        # 2. Define data directories
        bronze_lms_directory = "datamart/bronze/lms/"
        silver_loan_daily_directory = "datamart/silver/loan_daily/"
        gold_label_store_directory = "datamart/gold/label_store/"

        # Create directories if they do not exist
        os.makedirs(bronze_lms_directory, exist_ok=True)
        os.makedirs(silver_loan_daily_directory, exist_ok=True)
        os.makedirs(gold_label_store_directory, exist_ok=True)

        # 3. Bronze
        print("\n--- Step 1: Bronze ---\n")

        process_bronze_table(
            snapshotdate,
            bronze_lms_directory,
            spark
        )

        # 4. Silver
        print("\n--- Step 2: Silver ---\n")

        process_silver_table(
            snapshotdate,
            bronze_lms_directory,
            silver_loan_daily_directory,
            spark
        )

        # 5. Gold
        print("\n--- Step 3: Gold ---\n")

        process_labels_gold_table(
            snapshotdate,
            silver_loan_daily_directory,
            gold_label_store_directory,
            spark,
            dpd=30,
            mob=6
        )

        print("\n--- incremental pipeline completed ---\n")

    finally:
        # 6. Stop SparkSession
        spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run incremental Bronze -> Silver -> Gold pipeline"
    )

    parser.add_argument(
        "--snapshotdate",
        type=str,
        required=True,
        help="Snapshot date in YYYY-MM-DD format"
    )

    args = parser.parse_args()

    main(args.snapshotdate)