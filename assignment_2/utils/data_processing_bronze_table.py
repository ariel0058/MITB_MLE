"""Ingest the four raw sources into date-partitioned Bronze CSVs."""

from pathlib import Path

from utils.common import SOURCE_FILES, read_csv_strings, validate_keys


def process_bronze_table(source, data_directory, bronze_directory, spark):
    """Preserve raw fields and values"""
    input_path = Path(data_directory) / SOURCE_FILES[source]
    output_path = Path(bronze_directory) / source
    df = read_csv_strings(input_path, source, spark).cache()
    try:
        validate_keys(df, source)
        rows = df.count()
        dates = df.select("snapshot_date").distinct().count()
        (
            df.repartition("snapshot_date").write.mode("overwrite")
            .option("header", True)
            .option("escape", '"')
            .option("ignoreLeadingWhiteSpace", False)
            .option("ignoreTrailingWhiteSpace", False)
            .option("nullValue", "\u0000")
            .option("emptyValue", "")
            .partitionBy("snapshot_date")
            .csv(str(output_path))
        )
        print(f"Bronze {source}: {rows:,} rows, {dates} dates -> {output_path}")
        return df
    finally:
        df.unpersist()
