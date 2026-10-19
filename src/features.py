"""
Step 2: Build behavioral features for each transaction.

The main idea: a transaction looks suspicious when it doesn't match what
normally happens to the account receiving the money. In PaySim almost every
sender appears only once, but receivers appear about 5 times each, so the
behavior features describe the RECEIVER account (nameDest).

The golden rule: every feature uses only information from EARLIER hours.
The model must never see the future, or its test results will be fake.
Transactions in the same hour have no known order, so the current hour
is left out of every history feature.

What this script does:
  1. Reads the clean transactions from Step 1 (data/delta/transactions)
  2. Adds simple features: hour of day, transaction type, log amount
  3. Adds receiver history features over the previous 1, 6 and 24 hours
     and over all earlier hours
  4. Splits the data by time into train / valid / test (70% / 15% / 15%)
  5. Checks that no feature is missing
  6. Saves the result to data/delta/features and prints a summary

Run from the project root:
    uv run python src/features.py
"""

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from spark_utils import DELTA_DIR, build_spark

# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------
TRANSACTIONS_PATH = DELTA_DIR / "transactions"
FEATURES_PATH = DELTA_DIR / "features"

# History windows, in hours (PaySim's "step" column is one hour)
WINDOW_HOURS = [1, 6, 24]

# Value used when an account has no earlier history
NO_HISTORY = -1.0

# Time-based split: the oldest 70% of rows train the model,
# the next 15% tune it, and the newest 15% test it.
TRAIN_SHARE = 0.70
VALID_SHARE = 0.15

# The columns Step 3 will feed into the models
FEATURE_COLUMNS = (
    ["hour_of_day", "is_transfer", "log_amount"]
    + [f"dest_count_{h}h" for h in WINDOW_HOURS]
    + [f"dest_amount_{h}h" for h in WINDOW_HOURS]
    + [
        "dest_count_prior",
        "dest_mean_prior",
        "dest_first_seen",
        "amount_vs_dest_mean",
        "hours_since_dest_last",
    ]
)


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------
def add_basic_features(df: DataFrame) -> DataFrame:
    """Features that come from the transaction itself."""
    return (
        df
        # step 1 is the first hour, so hour_of_day runs 0-23
        .withColumn("hour_of_day", (F.col("step") - 1) % 24)
        # Transfers are about 4x riskier than cash-outs (see Step 1)
        .withColumn("is_transfer", (F.col("type") == "TRANSFER").cast("int"))
        # Amounts range from tiny to millions; log makes them easier to compare
        .withColumn("log_amount", F.log1p("amount"))
    )


def add_receiver_features(df: DataFrame) -> DataFrame:
    """History of the receiving account, using earlier hours only."""
    by_receiver = Window.partitionBy("nameDest").orderBy("step")

    # rangeBetween(-h, -1) = the previous h hours, NOT counting the current hour
    for h in WINDOW_HOURS:
        recent = by_receiver.rangeBetween(-h, -1)
        df = df.withColumn(
            f"dest_count_{h}h", F.count("amount").over(recent)
        ).withColumn(
            f"dest_amount_{h}h",
            F.coalesce(F.sum("amount").over(recent), F.lit(0.0)),
        )

    # Everything the receiver did in all earlier hours
    all_past = by_receiver.rangeBetween(Window.unboundedPreceding, -1)
    df = (
        df.withColumn("dest_count_prior", F.count("amount").over(all_past))
        .withColumn("_dest_mean", F.avg("amount").over(all_past))
        .withColumn("_dest_last_step", F.max("step").over(all_past))
    )

    has_history = F.col("dest_count_prior") > 0
    return (
        df.withColumn("dest_first_seen", (~has_history).cast("int"))
        .withColumn(
            "dest_mean_prior",
            F.when(has_history, F.col("_dest_mean")).otherwise(F.lit(NO_HISTORY)),
        )
        # How big is this amount compared to what the receiver usually gets?
        .withColumn(
            "amount_vs_dest_mean",
            F.when(
                has_history & (F.col("_dest_mean") > 0),
                F.col("amount") / F.col("_dest_mean"),
            ).otherwise(F.lit(NO_HISTORY)),
        )
        # How long since the receiver last got money?
        .withColumn(
            "hours_since_dest_last",
            F.when(
                has_history, (F.col("step") - F.col("_dest_last_step")).cast("double")
            ).otherwise(F.lit(NO_HISTORY)),
        )
        .drop("_dest_mean", "_dest_last_step")
    )


def add_time_split(df: DataFrame) -> tuple[DataFrame, int, int]:
    """Split by time so the model is always tested on the future."""
    train_end, valid_end = df.approxQuantile(
        "step", [TRAIN_SHARE, TRAIN_SHARE + VALID_SHARE], 0.001
    )
    train_end, valid_end = int(train_end), int(valid_end)

    split = (
        F.when(F.col("step") <= train_end, "train")
        .when(F.col("step") <= valid_end, "valid")
        .otherwise("test")
    )
    return df.withColumn("split", split), train_end, valid_end


# ---------------------------------------------------------------------------
# Checks and summary
# ---------------------------------------------------------------------------
def validate(df: DataFrame, expected_rows: int) -> None:
    """Make sure no rows were lost and no feature is missing."""
    rows = df.count()
    if rows != expected_rows:
        raise ValueError(f"Row count changed: expected {expected_rows:,}, got {rows:,}")

    null_counts = (
        df.select(
            [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in FEATURE_COLUMNS]
        )
        .first()
        .asDict()
    )
    missing = {c: n for c, n in null_counts.items() if n}
    if missing:
        raise ValueError(f"Features with missing values: {missing}")

    print("Feature checks passed.")


def print_summary(df: DataFrame, train_end: int, valid_end: int) -> None:
    print("\n" + "=" * 60)
    print("SUMMARY: data/delta/features")
    print("=" * 60)
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print(
        f"Time split: train = hours 1-{train_end}, "
        f"valid = {train_end + 1}-{valid_end}, test = {valid_end + 1}-744"
    )

    print("\nRows and fraud per split:")
    (
        df.groupBy("split")
        .agg(
            F.count("*").alias("rows"),
            F.sum("isFraud").alias("fraud"),
            F.round(F.avg("isFraud") * 100, 3).alias("fraud_pct"),
            F.min("step").alias("first_hour"),
            F.max("step").alias("last_hour"),
        )
        .orderBy("first_hour")
        .show()
    )

    # Do fraud and normal transactions look different? Medians resist outliers.
    print("Fraud vs normal (median, except where noted):")
    (
        df.groupBy("isFraud")
        .agg(
            F.round(F.avg("dest_first_seen") * 100, 1).alias("first_seen_pct"),
            F.percentile_approx("dest_count_prior", 0.5).alias("dest_count_prior"),
            F.percentile_approx("dest_count_24h", 0.5).alias("dest_count_24h"),
            F.round(F.percentile_approx("amount", 0.5), 0).alias("amount"),
            F.round(F.percentile_approx("amount_vs_dest_mean", 0.5), 2).alias(
                "amount_vs_mean"
            ),
            F.round(F.avg("hour_of_day"), 1).alias("avg_hour"),
        )
        .orderBy("isFraud")
        .show()
    )

    print("Fraud rate by hour of day:")
    (
        df.groupBy("hour_of_day")
        .agg(
            F.count("*").alias("rows"),
            F.round(F.avg("isFraud") * 100, 2).alias("fraud_pct"),
        )
        .orderBy("hour_of_day")
        .show(24)
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    spark = build_spark("paysim-features")

    print(f"Reading {TRANSACTIONS_PATH} ...")
    transactions = spark.read.format("delta").load(TRANSACTIONS_PATH.as_posix())
    input_rows = transactions.count()
    print(f"Input rows: {input_rows:,}")

    features = add_basic_features(transactions)
    features = add_receiver_features(features)
    features, train_end, valid_end = add_time_split(features)

    features = features.select(
        "step",
        "type",
        "amount",
        "nameOrig",
        "nameDest",
        *FEATURE_COLUMNS,
        "isFlaggedFraud",
        "isFraud",
        "split",
    )

    print(f"Writing Delta table to {FEATURES_PATH} ...")
    (
        features.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .partitionBy("split")
        .save(FEATURES_PATH.as_posix())
    )

    saved = spark.read.format("delta").load(FEATURES_PATH.as_posix())
    validate(saved, input_rows)
    print_summary(saved, train_end, valid_end)

    spark.stop()
    print("Done.")


if __name__ == "__main__":
    main()
