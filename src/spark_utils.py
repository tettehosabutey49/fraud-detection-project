"""
Shared Spark setup used by every step of the pipeline.

Keeping this in one place means ingest.py, features.py and later scripts
all run Spark with the same settings.
"""

import os
import sys
from pathlib import Path

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DELTA_DIR = PROJECT_ROOT / "data" / "delta"

# On Windows, Spark needs winutils.exe and hadoop.dll to write files.
# Put them in C:\hadoop\bin, or set HADOOP_HOME to another folder.
WINDOWS_HADOOP_HOME = Path(os.environ.get("HADOOP_HOME", r"C:\hadoop"))


def configure_windows() -> None:
    """Point Spark at winutils on Windows. Does nothing on macOS/Linux."""
    if os.name != "nt":
        return

    winutils = WINDOWS_HADOOP_HOME / "bin" / "winutils.exe"
    if not winutils.exists():
        raise FileNotFoundError(
            f"winutils.exe not found at {winutils}.\n"
            "Download winutils.exe and hadoop.dll into that folder "
            "(see the Windows setup notes in the README)."
        )

    os.environ["HADOOP_HOME"] = str(WINDOWS_HADOOP_HOME)
    os.environ["PATH"] = (
        str(WINDOWS_HADOOP_HOME / "bin") + os.pathsep + os.environ.get("PATH", "")
    )


def build_spark(app_name: str) -> SparkSession:
    """Create a local Spark session with Delta Lake enabled."""
    configure_windows()

    # Make Spark workers use this project's Python (avoids errors on Windows)
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    builder = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        # The default of 200 shuffle partitions is built for clusters;
        # a smaller number is faster on a single laptop.
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark
