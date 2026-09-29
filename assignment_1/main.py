"""Run the Assignment 1 Bronze, Silver, and Gold pipeline."""

import argparse
from pathlib import Path

from pyspark.sql import SparkSession

from utils.common import SOURCE_FILES
from utils.data_processing_bronze_table import process_bronze_table
from utils.data_processing_gold_table import process_gold_tables
from utils.data_processing_silver_table import process_silver_table


def main(stage="all"):
    root = Path(__file__).resolve().parent
    data_directory = root / "data"
    bronze_directory = root / "datamart" / "bronze"
    silver_directory = root / "datamart" / "silver"
    gold_directory = root / "datamart" / "gold"
    spark = (
        SparkSession.builder.appName("assignment1-medallion-pipeline")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    try:
        if stage in ("bronze", "all"):
            for source in SOURCE_FILES:
                process_bronze_table(source, data_directory, bronze_directory, spark)
        if stage in ("silver", "all"):
            for source in SOURCE_FILES:
                process_silver_table(source, bronze_directory, silver_directory, spark)
        if stage in ("gold", "all"):
            process_gold_tables(silver_directory, gold_directory, spark)
        print(f"Completed {stage} pipeline stage. Original CSVs are unchanged.")
    finally:
        spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Bronze, Silver, and Gold tables")
    parser.add_argument("--stage", choices=["all", "bronze", "silver", "gold"], default="all")
    main(parser.parse_args().stage)
