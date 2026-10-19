"""
Step 1: Ingest the PaySim dataset into Delta Lake.

What this script does:
  1. Reads the raw PaySim CSV with an explicit schema
  2. Keeps only TRANSFER and CASH_OUT transactions (the only types with fraud)
  3. Drops the balance columns that leak the fraud label
  4. Runs basic data quality checks
  5. Saves the result as a Delta table at data/delta/transactions
  6. Prints a short summary of the saved data

Run from the project root:
    uv run python src/ingest.py
"""
"""
Step 1: Ingest the PaySim dataset into Delta Lake.

What this script does:
  1. Reads the raw PaySim CSV with an explicit schema
  2. Keeps only TRANSFER and CASH_OUT transactions (the only types with fraud)
  3. Drops the balance columns that leak the fraud label
  4. Runs basic data quality checks
  5. Saves the result as a Delta table at data/delta/transactions
  6. Prints a short summary of the saved data

Run from the project root:
    uv run python src/ingest.py
"""

from pathlib import Path
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from spark_utils import DELTA_DIR, PROJECT_ROOT, build_spark

# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DELTA_PATH = DELTA_DIR / "transactions"

# Fraud only occurs in these transaction types
FRAUD_TYPES = ["TRANSFER", "CASH_OUT"]

# Fraudulent transactions are cancelled in PaySim, so these columns
# reveal the label. Using them would make the model look unrealistically good.
LEAKY_COLUMNS = [
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

# Explicit schema: faster than inferSchema and protects against type surprises
PAYSIM_SCHEMA = StructType(
    [
        StructField("step", IntegerType(), nullable=False),
        StructField("type", StringType(), nullable=False),
        StructField("amount", DoubleType(), nullable=False),
        StructField("nameOrig", StringType(), nullable=False),
        StructField("oldbalanceOrg", DoubleType(), nullable=True),
        StructField("newbalanceOrig", DoubleType(), nullable=True),
        StructField("nameDest", StringType(), nullable=False),
        StructField("oldbalanceDest", DoubleType(), nullable=True),
        StructField("newbalanceDest", DoubleType(), nullable=True),
        StructField("isFraud", IntegerType(), nullable=False),
        StructField("isFlaggedFraud", IntegerType(), nullable=False),
    ]
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_raw_csv() -> Path:
    """Find the PaySim CSV in data/raw."""
    files = sorted(RAW_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSV found in {RAW_DIR}.\n"
            "Download it with:\n"
            "  uvx kaggle datasets download -d ealaxi/paysim1 -p data/raw --unzip"
        )
    return files[0]


def validate(df: DataFrame) -> None:
    """Basic data quality checks. Raises an error if any check fails."""
    checks = {
        "null values in key columns": df.filter(
            F.col("step").isNull()
            | F.col("type").isNull()
            | F.col("amount").isNull()
            | F.col("nameOrig").isNull()
            | F.col("nameDest").isNull()
            | F.col("isFraud").isNull()
        ).count(),
        "negative amounts": df.filter(F.col("amount") < 0).count(),
        "steps outside 1-744": df.filter(~F.col("step").between(1, 744)).count(),
        "isFraud not 0 or 1": df.filter(~F.col("isFraud").isin(0, 1)).count(),
        "unexpected transaction types": df.filter(
            ~F.col("type").isin(FRAUD_TYPES)
        ).count(),
    }

    failed = {name: n for name, n in checks.items() if n > 0}
    if failed:
        raise ValueError(f"Data quality checks failed: {failed}")
    print("Data quality checks passed.")


def print_summary(df: DataFrame) -> None:
    """Print a quick overview of the saved table."""
    total = df.count()
    fraud = df.filter(F.col("isFraud") == 1).count()

    print("\n" + "=" * 50)
    print("SUMMARY: data/delta/transactions")
    print("=" * 50)
    print(f"Rows:            {total:,}")
    print(f"Fraud rows:      {fraud:,} ({fraud / total:.3%})")

    print("\nFraud by transaction type:")
    (
        df.groupBy("type")
        .agg(
            F.count("*").alias("rows"),
            F.sum("isFraud").alias("fraud"),
            F.round(F.avg("isFraud") * 100, 3).alias("fraud_pct"),
        )
        .orderBy("type")
        .show()
    )

    # How often do accounts repeat? This decides how we model behavior in Step 2.
    print("Account activity:")
    (
        df.agg(
            F.countDistinct("nameOrig").alias("unique_senders"),
            F.countDistinct("nameDest").alias("unique_receivers"),
        ).show()
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    spark = build_spark("paysim-ingest")

    csv_path = find_raw_csv()
    print(f"Reading {csv_path.name} ...")
    raw = spark.read.csv(str(csv_path), header=True, schema=PAYSIM_SCHEMA)
    raw_count = raw.count()

    transactions = (
        raw.filter(F.col("type").isin(FRAUD_TYPES))
        .drop(*LEAKY_COLUMNS)
        .withColumn("source_file", F.lit(csv_path.name))
        .withColumn("ingested_at", F.current_timestamp())
    )
    kept_count = transactions.count()
    print(f"Raw rows: {raw_count:,} | Kept (TRANSFER + CASH_OUT): {kept_count:,}")

    validate(transactions)

    print(f"Writing Delta table to {DELTA_PATH} ...")
    (
        transactions.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .partitionBy("type")
        .save(DELTA_PATH.as_posix())
    )

    # Read back from Delta to confirm the write worked
    saved = spark.read.format("delta").load(DELTA_PATH.as_posix())
    print_summary(saved)

    spark.stop()
    print("Done.")


if __name__ == "__main__":
    main()