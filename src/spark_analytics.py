"""PySpark sales analytics for synthetic e-commerce Parquet datasets."""

from __future__ import annotations

import logging
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SalesAnalytics:
    """Run sales analytics over e-commerce order and product datasets.

    Args:
        app_name: Spark application name.
    """

    def __init__(self, app_name: str = config.APP_NAME) -> None:
        """Initialize the analytics class without starting Spark immediately."""
        self.app_name = app_name
        self.spark: SparkSession | None = None

    def create_spark_session(self) -> SparkSession:
        """Create a local SparkSession configured for analytics workloads.

        Configuration includes local execution, 4GB driver/executor memory,
        adaptive query execution, and Kryo serialization.

        Returns:
            A configured SparkSession instance.
        """
        logger.info("Creating Spark session: %s", self.app_name)
        self.spark = (
            SparkSession.builder.appName(self.app_name)
            .master("local[*]")
            .config("spark.driver.memory", "4g")
            .config("spark.executor.memory", "4g")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .config("spark.sql.adaptive.skewJoin.enabled", "true")
            .config("spark.sql.shuffle.partitions", "8")
            .getOrCreate()
        )
        return self.spark

    def load_parquet(self, path: str | Path) -> DataFrame:
        """Load a Parquet dataset into a Spark DataFrame.

        Args:
            path: Path to a Parquet file or directory.

        Returns:
            A Spark DataFrame loaded from the Parquet path.
        """
        spark = self._get_or_create_spark()
        parquet_path = Path(path)
        logger.info("Loading Parquet dataset from %s", parquet_path)
        return spark.read.parquet(str(parquet_path))

    def top_customers_by_revenue(
        self,
        orders_df: DataFrame,
        products_df: DataFrame,
        n: int = 10,
    ) -> DataFrame:
        """Calculate the top customers by total revenue.

        Revenue is calculated as order quantity multiplied by product price.

        Args:
            orders_df: Orders DataFrame containing customer_id, product_id, and quantity.
            products_df: Products DataFrame containing product_id and price.
            n: Number of customers to return.

        Returns:
            A Spark DataFrame with customer_id, total_revenue, total_orders,
            and total_units, sorted by revenue descending.
        """
        logger.info("Calculating top %s customers by revenue", n)
        enriched_orders = self._orders_with_revenue(orders_df, products_df)

        return (
            enriched_orders.groupBy("customer_id")
            .agg(
                F.round(F.sum("revenue"), 2).alias("total_revenue"),
                F.countDistinct("order_id").alias("total_orders"),
                F.sum("quantity").alias("total_units"),
            )
            .orderBy(F.desc("total_revenue"))
            .limit(n)
        )

    def sales_by_category(self, orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """Calculate revenue and units sold by product category.

        Args:
            orders_df: Orders DataFrame containing product_id and quantity.
            products_df: Products DataFrame containing product_id, category, and price.

        Returns:
            A Spark DataFrame with category-level revenue, order count, and units sold.
        """
        logger.info("Calculating sales by category")
        enriched_orders = self._orders_with_revenue(orders_df, products_df)

        return (
            enriched_orders.groupBy("category")
            .agg(
                F.round(F.sum("revenue"), 2).alias("total_revenue"),
                F.sum("quantity").alias("units_sold"),
                F.countDistinct("order_id").alias("order_count"),
            )
            .orderBy(F.desc("total_revenue"))
        )

    def monthly_trends(self, orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """Calculate monthly revenue and month-over-month growth percentage.

        Args:
            orders_df: Orders DataFrame containing order_date, product_id, and quantity.
            products_df: Products DataFrame containing product_id and price.

        Returns:
            A Spark DataFrame with order_month, monthly_revenue,
            previous_month_revenue, and revenue_growth_pct.
        """
        logger.info("Calculating monthly revenue trends")
        enriched_orders = self._orders_with_revenue(orders_df, products_df)
        month_window = Window.orderBy("order_month")

        monthly_revenue = (
            enriched_orders.withColumn("order_month", F.date_format(F.to_date("order_date"), "yyyy-MM"))
            .groupBy("order_month")
            .agg(
                F.round(F.sum("revenue"), 2).alias("monthly_revenue"),
                F.countDistinct("order_id").alias("order_count"),
                F.sum("quantity").alias("units_sold"),
            )
        )

        return (
            monthly_revenue.withColumn(
                "previous_month_revenue",
                F.lag("monthly_revenue").over(month_window),
            )
            .withColumn(
                "revenue_growth_pct",
                F.when(
                    F.col("previous_month_revenue").isNull()
                    | (F.col("previous_month_revenue") == 0),
                    F.lit(None),
                ).otherwise(
                    F.round(
                        ((F.col("monthly_revenue") - F.col("previous_month_revenue"))
                         / F.col("previous_month_revenue"))
                        * 100,
                        2,
                    )
                ),
            )
            .orderBy("order_month")
        )

    def stop(self) -> None:
        """Stop the active SparkSession if one exists."""
        if self.spark is not None:
            logger.info("Stopping Spark session")
            self.spark.stop()
            self.spark = None

    def _get_or_create_spark(self) -> SparkSession:
        """Return the active SparkSession or create one if needed."""
        if self.spark is None:
            return self.create_spark_session()
        return self.spark

    @staticmethod
    def _orders_with_revenue(orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """Join orders with products and add a revenue column.

        Args:
            orders_df: Orders DataFrame with product_id and quantity.
            products_df: Products DataFrame with product_id, category, and price.

        Returns:
            A joined Spark DataFrame containing revenue per order row.
        """
        product_columns = products_df.select("product_id", "category", "price")
        return (
            orders_df.join(product_columns, on="product_id", how="inner")
            .withColumn("quantity", F.col("quantity").cast("double"))
            .withColumn("price", F.col("price").cast("double"))
            .withColumn("revenue", F.col("quantity") * F.col("price"))
        )


def run_example() -> None:
    """Run analytics against Parquet files in the configured raw data directory."""
    analytics = SalesAnalytics()
    try:
        orders = analytics.load_parquet(config.RAW_DATA_DIR / "orders.parquet")
        products = analytics.load_parquet(config.RAW_DATA_DIR / "products.parquet")

        analytics.top_customers_by_revenue(orders, products).show(truncate=False)
        analytics.sales_by_category(orders, products).show(truncate=False)
        analytics.monthly_trends(orders, products).show(truncate=False)
    finally:
        analytics.stop()


if __name__ == "__main__":
    run_example()
