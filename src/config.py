"""Project configuration values for the e-commerce data pipeline."""

from pathlib import Path


BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

CUSTOMERS_FILE: Path = RAW_DATA_DIR / "customers.csv"
PRODUCTS_FILE: Path = RAW_DATA_DIR / "products.csv"
ORDERS_FILE: Path = RAW_DATA_DIR / "orders.csv"

DEFAULT_CUSTOMER_COUNT: int = 1_000
DEFAULT_PRODUCT_COUNT: int = 100
DEFAULT_ORDER_COUNT: int = 5_000
DEFAULT_RANDOM_SEED: int = 42

APP_NAME: str = "ECommerceDataPipeline"
