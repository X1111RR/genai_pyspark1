"""Compare Pandas and PySpark performance on e-commerce Parquet data.

The benchmark loads orders.parquet and products.parquet, joins on product_id,
calculates revenue, aggregates revenue by customer_id, and returns the top 10
customers for both Pandas and PySpark.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src import config
from src.spark_analytics import SalesAnalytics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class TimedResult:
    """Store the result and elapsed time for a measured operation."""

    value: object
    elapsed_seconds: float


@dataclass(frozen=True)
class BenchmarkSummary:
    """Store per-stage benchmark timings for one execution engine."""

    engine: str
    load_seconds: float
    join_and_revenue_seconds: float
    group_seconds: float
    top_10_seconds: float

    @property
    def total_seconds(self) -> float:
        """Return total elapsed benchmark time in seconds."""
        return (
            self.load_seconds
            + self.join_and_revenue_seconds
            + self.group_seconds
            + self.top_10_seconds
        )


def time_operation(operation: Callable[[], T]) -> TimedResult:
    """Run and time one operation.

    Args:
        operation: Callable to execute.

    Returns:
        TimedResult containing the operation result and elapsed wall-clock time.
    """
    start_time = time.perf_counter()
    value = operation()
    elapsed_seconds = time.perf_counter() - start_time
    return TimedResult(value=value, elapsed_seconds=elapsed_seconds)


def benchmark_pandas(orders_path: Path, products_path: Path) -> tuple[BenchmarkSummary, pd.DataFrame]:
    """Benchmark the analytics workflow using Pandas.

    Args:
        orders_path: Path to orders.parquet.
        products_path: Path to products.parquet.

    Returns:
        A tuple of benchmark summary and top-10 customers DataFrame.
    """
    logger.info("Starting Pandas benchmark")

    load_result = time_operation(
        lambda: (
            pd.read_parquet(orders_path),
            pd.read_parquet(products_path),
        )
    )
    orders_df, products_df = load_result.value  # type: ignore[misc]

    join_result = time_operation(
        lambda: orders_df.merge(
            products_df[["product_id", "price"]],
            on="product_id",
            how="inner",
        ).assign(revenue=lambda dataframe: dataframe["quantity"] * dataframe["price"])
    )
    joined_df = join_result.value

    group_result = time_operation(
        lambda: joined_df.groupby("customer_id", as_index=False)["revenue"].sum()
    )
    revenue_by_customer = group_result.value

    top_result = time_operation(
        lambda: revenue_by_customer.nlargest(10, "revenue").reset_index(drop=True)
    )
    top_customers = top_result.value

    summary = BenchmarkSummary(
        engine="Pandas",
        load_seconds=load_result.elapsed_seconds,
        join_and_revenue_seconds=join_result.elapsed_seconds,
        group_seconds=group_result.elapsed_seconds,
        top_10_seconds=top_result.elapsed_seconds,
    )
    return summary, top_customers


def benchmark_pyspark(
    spark: SparkSession,
    orders_path: Path,
    products_path: Path,
) -> tuple[BenchmarkSummary, pd.DataFrame]:
    """Benchmark the analytics workflow using PySpark.

    Spark transformations are lazy, so each stage is materialized with an action
    to make the stage timing meaningful.

    Args:
        spark: Active SparkSession.
        orders_path: Path to orders.parquet.
        products_path: Path to products.parquet.

    Returns:
        A tuple of benchmark summary and top-10 customers as a pandas DataFrame.
    """
    logger.info("Starting PySpark benchmark")

    def load_dataframes() -> tuple[DataFrame, DataFrame]:
        orders = spark.read.parquet(str(orders_path)).cache()
        products = spark.read.parquet(str(products_path)).cache()
        orders.count()
        products.count()
        return orders, products

    load_result = time_operation(load_dataframes)
    orders_df, products_df = load_result.value  # type: ignore[misc]

    def join_and_calculate_revenue() -> DataFrame:
        joined = (
            orders_df.join(products_df.select("product_id", "price"), on="product_id", how="inner")
            .withColumn("revenue", F.col("quantity").cast("double") * F.col("price").cast("double"))
            .cache()
        )
        joined.count()
        return joined

    join_result = time_operation(join_and_calculate_revenue)
    joined_df = join_result.value

    def group_revenue() -> DataFrame:
        grouped = (
            joined_df.groupBy("customer_id")
            .agg(F.round(F.sum("revenue"), 2).alias("revenue"))
            .cache()
        )
        grouped.count()
        return grouped

    group_result = time_operation(group_revenue)
    revenue_by_customer = group_result.value

    top_result = time_operation(
        lambda: revenue_by_customer.orderBy(F.desc("revenue")).limit(10).toPandas()
    )
    top_customers = top_result.value

    summary = BenchmarkSummary(
        engine="PySpark",
        load_seconds=load_result.elapsed_seconds,
        join_and_revenue_seconds=join_result.elapsed_seconds,
        group_seconds=group_result.elapsed_seconds,
        top_10_seconds=top_result.elapsed_seconds,
    )
    return summary, top_customers


def build_comparison_table(summaries: list[BenchmarkSummary]) -> pd.DataFrame:
    """Build a formatted comparison table from benchmark summaries."""
    rows = []
    for summary in summaries:
        rows.append(
            {
                "Engine": summary.engine,
                "Load Time Sec": summary.load_seconds,
                "Join + Revenue Sec": summary.join_and_revenue_seconds,
                "GroupBy Sec": summary.group_seconds,
                "Top 10 Sec": summary.top_10_seconds,
                "Total Sec": summary.total_seconds,
            }
        )

    comparison = pd.DataFrame(rows)
    numeric_columns = comparison.select_dtypes(include=["float64", "float32"]).columns
    comparison[numeric_columns] = comparison[numeric_columns].round(3)
    return comparison


def run_comparison() -> None:
    """Run Pandas and PySpark benchmarks and print results."""
    orders_path = config.RAW_DATA_DIR / "orders.parquet"
    products_path = config.RAW_DATA_DIR / "products.parquet"

    if not orders_path.exists() or not products_path.exists():
        raise FileNotFoundError(
            "Missing Parquet input files. Run `python main.py` first to create "
            "data/raw/orders.parquet and data/raw/products.parquet."
        )

    pandas_summary, pandas_top_customers = benchmark_pandas(orders_path, products_path)

    analytics = SalesAnalytics(app_name="PandasVsPySparkBenchmark")
    spark = analytics.create_spark_session()
    try:
        spark_summary, spark_top_customers = benchmark_pyspark(spark, orders_path, products_path)
    finally:
        analytics.stop()

    comparison = build_comparison_table([pandas_summary, spark_summary])

    print("\nPandas vs PySpark Performance Comparison")
    print("=" * 92)
    print(comparison.to_string(index=False))
    print("=" * 92)

    print("\nPandas Top 10 Customers")
    print(pandas_top_customers.to_string(index=False))

    print("\nPySpark Top 10 Customers")
    print(spark_top_customers.to_string(index=False))


if __name__ == "__main__":
    run_comparison()
