"""Benchmark common file formats with timing, memory, CPU, and energy metrics.

Formats benchmarked:
- CSV
- XLSX
- Parquet
- ORC
- Feather

The benchmark creates a 500,000-row synthetic DataFrame, writes and reads each
format, then prints a comparison table and percentage savings versus CSV.
"""

from __future__ import annotations

import importlib.resources
import logging
import os
import shutil
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

if not os.environ.get("TZDIR"):
    try:
        os.environ["TZDIR"] = str(importlib.resources.files("tzdata") / "zoneinfo")
    except ModuleNotFoundError:
        pass

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.orc as orc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ROW_COUNT: int = 500_000
TDP_WATTS: float = 65.0
OUTPUT_DIR: Path = Path("data") / "benchmarks"


@dataclass(frozen=True)
class BenchmarkResult:
    """Stores measured benchmark metrics for one file format."""

    format_name: str
    file_path: Path
    file_size_mb: float
    write_time_seconds: float
    read_time_seconds: float
    peak_memory_mb: float
    cpu_time_seconds: float
    estimated_energy_wh: float


@dataclass(frozen=True)
class OperationMetrics:
    """Stores timing and resource metrics for one operation."""

    wall_time_seconds: float
    cpu_time_seconds: float
    peak_memory_mb: float


Writer = Callable[[pd.DataFrame, Path], None]
Reader = Callable[[Path], pd.DataFrame]


def create_dataframe(row_count: int = ROW_COUNT) -> pd.DataFrame:
    """Create a synthetic benchmark DataFrame.

    Args:
        row_count: Number of rows to generate.

    Returns:
        A pandas DataFrame with id, name, email, amount, date, and category.
    """
    logger.info("Creating DataFrame with %s rows", row_count)
    rng = np.random.default_rng(42)
    ids = np.arange(1, row_count + 1)
    categories = np.array(["Electronics", "Clothing", "Home", "Sports", "Books"])
    dates = pd.date_range("2024-01-01", periods=row_count, freq="min").strftime("%Y-%m-%d")

    dataframe = pd.DataFrame(
        {
            "id": ids,
            "name": [f"Customer {value}" for value in ids],
            "email": [f"customer{value}@example.com" for value in ids],
            "amount": rng.uniform(10, 5_000, size=row_count).round(2),
            "date": dates,
            "category": rng.choice(categories, size=row_count),
        }
    )
    string_columns = ["name", "email", "date", "category"]
    dataframe[string_columns] = dataframe[string_columns].astype("object")
    return dataframe


def measure_operation(operation: Callable[[], object]) -> OperationMetrics:
    """Measure wall time, CPU time, and peak memory for one operation.

    Args:
        operation: Callable operation to execute.

    Returns:
        OperationMetrics for the operation.
    """
    tracemalloc.start()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()

    operation()

    cpu_time = time.process_time() - cpu_start
    wall_time = time.perf_counter() - wall_start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return OperationMetrics(
        wall_time_seconds=wall_time,
        cpu_time_seconds=cpu_time,
        peak_memory_mb=peak_bytes / (1024 * 1024),
    )


def file_size_mb(path: Path) -> float:
    """Calculate a file's size in megabytes."""
    return path.stat().st_size / (1024 * 1024)


def estimate_energy_wh(cpu_time_seconds: float) -> float:
    """Estimate energy consumption using CPU time and fixed TDP.

    Formula:
        CPU_time * 65W TDP / 3600 = Wh
    """
    return cpu_time_seconds * TDP_WATTS / 3600


def write_csv(dataframe: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame as CSV."""
    dataframe.to_csv(path, index=False)


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file into a DataFrame."""
    return pd.read_csv(path)


def write_xlsx(dataframe: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame as XLSX using openpyxl."""
    dataframe.to_excel(path, index=False, engine="openpyxl")


def read_xlsx(path: Path) -> pd.DataFrame:
    """Read an XLSX file into a DataFrame using openpyxl."""
    return pd.read_excel(path, engine="openpyxl")


def write_parquet(dataframe: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame as Parquet using fastparquet."""
    dataframe.to_parquet(path, index=False, engine="fastparquet", compression="snappy")


def read_parquet(path: Path) -> pd.DataFrame:
    """Read a Parquet file into a DataFrame using fastparquet."""
    return pd.read_parquet(path, engine="fastparquet")


def write_orc(dataframe: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame as ORC using pyarrow."""
    table = pa.Table.from_pandas(dataframe, preserve_index=False)
    with path.open("wb") as output_file:
        orc.write_table(table, output_file)


def read_orc(path: Path) -> pd.DataFrame:
    """Read an ORC file into a DataFrame using pyarrow."""
    with path.open("rb") as input_file:
        table = orc.ORCFile(input_file).read()
    return table.to_pandas()


def write_feather(dataframe: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame as Feather using pyarrow."""
    table = pa.Table.from_pandas(dataframe, preserve_index=False)
    feather.write_feather(table, path, compression="zstd")


def read_feather(path: Path) -> pd.DataFrame:
    """Read a Feather file into a DataFrame using pyarrow."""
    return feather.read_feather(path)


def benchmark_format(
    dataframe: pd.DataFrame,
    format_name: str,
    file_path: Path,
    writer: Writer,
    reader: Reader,
) -> BenchmarkResult:
    """Benchmark writing and reading one file format.

    Args:
        dataframe: Source DataFrame to benchmark.
        format_name: Name of the file format.
        file_path: Output file path.
        writer: Function that writes the DataFrame to file_path.
        reader: Function that reads file_path into a DataFrame.

    Returns:
        BenchmarkResult containing measured metrics.
    """
    logger.info("Benchmarking %s", format_name)
    write_metrics = measure_operation(lambda: writer(dataframe, file_path))
    read_metrics = measure_operation(lambda: reader(file_path))
    total_cpu_time = write_metrics.cpu_time_seconds + read_metrics.cpu_time_seconds

    return BenchmarkResult(
        format_name=format_name,
        file_path=file_path,
        file_size_mb=file_size_mb(file_path),
        write_time_seconds=write_metrics.wall_time_seconds,
        read_time_seconds=read_metrics.wall_time_seconds,
        peak_memory_mb=max(write_metrics.peak_memory_mb, read_metrics.peak_memory_mb),
        cpu_time_seconds=total_cpu_time,
        estimated_energy_wh=estimate_energy_wh(total_cpu_time),
    )


def percentage_savings(value: float, baseline: float) -> float:
    """Calculate percentage savings versus a baseline value."""
    if baseline == 0:
        return 0.0
    return ((baseline - value) / baseline) * 100


def results_to_dataframe(results: list[BenchmarkResult]) -> pd.DataFrame:
    """Convert benchmark results to a formatted comparison DataFrame."""
    baseline = next(result for result in results if result.format_name == "CSV")
    rows: list[dict[str, float | str]] = []

    for result in results:
        rows.append(
            {
                "Format": result.format_name,
                "File Size MB": result.file_size_mb,
                "Write Time Sec": result.write_time_seconds,
                "Read Time Sec": result.read_time_seconds,
                "Peak Memory MB": result.peak_memory_mb,
                "CPU Time Sec": result.cpu_time_seconds,
                "Energy Wh": result.estimated_energy_wh,
                "Size Savings vs CSV %": percentage_savings(result.file_size_mb, baseline.file_size_mb),
                "Write Savings vs CSV %": percentage_savings(
                    result.write_time_seconds,
                    baseline.write_time_seconds,
                ),
                "Read Savings vs CSV %": percentage_savings(
                    result.read_time_seconds,
                    baseline.read_time_seconds,
                ),
                "Energy Savings vs CSV %": percentage_savings(
                    result.estimated_energy_wh,
                    baseline.estimated_energy_wh,
                ),
            }
        )

    return pd.DataFrame(rows)


def print_results(results: list[BenchmarkResult]) -> None:
    """Print a formatted benchmark comparison table."""
    comparison = results_to_dataframe(results)
    numeric_columns = comparison.select_dtypes(include=["float64", "float32"]).columns
    comparison[numeric_columns] = comparison[numeric_columns].round(2)

    print("\nFile Format Benchmark Results")
    print("=" * 160)
    print(comparison.to_string(index=False))
    print("=" * 160)


def run_benchmark() -> None:
    """Run all file format benchmarks and print results."""
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataframe = create_dataframe()
    benchmarks: list[tuple[str, Path, Writer, Reader]] = [
        ("CSV", OUTPUT_DIR / "benchmark.csv", write_csv, read_csv),
        ("XLSX", OUTPUT_DIR / "benchmark.xlsx", write_xlsx, read_xlsx),
        ("Parquet", OUTPUT_DIR / "benchmark.parquet", write_parquet, read_parquet),
        ("ORC", OUTPUT_DIR / "benchmark.orc", write_orc, read_orc),
        ("Feather", OUTPUT_DIR / "benchmark.feather", write_feather, read_feather),
    ]

    results: list[BenchmarkResult] = []
    for format_name, file_path, writer, reader in benchmarks:
        try:
            results.append(benchmark_format(dataframe, format_name, file_path, writer, reader))
        except Exception:
            logger.exception("Failed to benchmark %s", format_name)
            raise

    print_results(results)


if __name__ == "__main__":
    run_benchmark()
