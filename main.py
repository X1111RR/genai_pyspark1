"""Run the full e-commerce synthetic data pipeline.

This script generates customers, products, and orders with SyntheticDataGenerator
and saves each dataset as a Parquet file in data/raw/.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

import pandas as pd

from src import config
from src.data_generator import SyntheticDataGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


DatasetFactory = Callable[[], pd.DataFrame]


def format_duration(seconds: float) -> str:
    """Format elapsed seconds as a compact human-readable string.

    Args:
        seconds: Elapsed time in seconds.

    Returns:
        A formatted duration string.
    """
    minutes, remaining_seconds = divmod(seconds, 60)
    hours, remaining_minutes = divmod(minutes, 60)

    if hours >= 1:
        return f"{int(hours)}h {int(remaining_minutes)}m {remaining_seconds:.2f}s"
    if minutes >= 1:
        return f"{int(minutes)}m {remaining_seconds:.2f}s"
    return f"{seconds:.2f}s"


def format_file_size(file_path: Path) -> str:
    """Return a readable file size for a saved file.

    Args:
        file_path: Path to the file whose size should be displayed.

    Returns:
        File size formatted in B, KB, MB, or GB.
    """
    size_bytes = file_path.stat().st_size
    units = ["B", "KB", "MB", "GB"]
    size = float(size_bytes)

    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024

    return f"{size_bytes} B"


def save_parquet(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Save a pandas DataFrame as a Parquet file.

    Args:
        dataframe: DataFrame to save.
        output_path: Destination Parquet file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(output_path, index=False, engine="pyarrow", compression="snappy")


def generate_and_save_dataset(
    dataset_name: str,
    dataset_factory: DatasetFactory,
    output_path: Path,
) -> float:
    """Generate one dataset, save it to Parquet, and log timing details.

    Args:
        dataset_name: Human-readable dataset name for logs.
        dataset_factory: Callable that returns the generated DataFrame.
        output_path: Destination Parquet file path.

    Returns:
        Elapsed time in seconds for generation plus saving.
    """
    logger.info("Starting %s generation", dataset_name)
    start_time = time.perf_counter()

    dataframe = dataset_factory()
    generation_time = time.perf_counter() - start_time
    logger.info("Generated %s rows for %s", len(dataframe), dataset_name)
    print(f"{dataset_name} generation time: {format_duration(generation_time)}")

    logger.info("Saving %s to %s", dataset_name, output_path)
    save_start_time = time.perf_counter()
    save_parquet(dataframe, output_path)
    save_time = time.perf_counter() - save_start_time

    file_size = format_file_size(output_path)
    elapsed_time = time.perf_counter() - start_time
    logger.info(
        "Saved %s to %s in %s; file size=%s",
        dataset_name,
        output_path,
        format_duration(save_time),
        file_size,
    )
    print(f"{dataset_name} save time: {format_duration(save_time)}")
    print(f"{dataset_name} file size: {file_size}")
    print(f"{dataset_name} total time: {format_duration(elapsed_time)}")

    return elapsed_time


def run_pipeline() -> None:
    """Run the full synthetic e-commerce data pipeline."""
    total_start_time = time.perf_counter()
    logger.info("Starting e-commerce data pipeline")

    generator = SyntheticDataGenerator(
        customer_count=100_000,
        product_count=10_000,
        order_count=1_000_000,
        seed=config.DEFAULT_RANDOM_SEED,
    )

    outputs = {
        "customers": config.RAW_DATA_DIR / "customers.parquet",
        "products": config.RAW_DATA_DIR / "products.parquet",
        "orders": config.RAW_DATA_DIR / "orders.parquet",
    }

    try:
        generate_and_save_dataset("customers", generator.generate_customers, outputs["customers"])
        generate_and_save_dataset("products", generator.generate_products, outputs["products"])
        generate_and_save_dataset("orders", generator.generate_orders, outputs["orders"])
    except Exception:
        logger.exception("Pipeline failed")
        raise
    finally:
        total_elapsed_time = time.perf_counter() - total_start_time
        logger.info("Pipeline finished in %s", format_duration(total_elapsed_time))
        print(f"Total execution time: {format_duration(total_elapsed_time)}")


if __name__ == "__main__":
    run_pipeline()
