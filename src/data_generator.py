"""Generate synthetic e-commerce datasets for local testing.

The main API is the SyntheticDataGenerator class. It creates customers,
products, and orders as pandas DataFrames and can optionally save them as CSVs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker
from numpy.random import Generator
from tqdm.auto import tqdm

from src.config import (
    CUSTOMERS_FILE,
    DEFAULT_CUSTOMER_COUNT,
    DEFAULT_ORDER_COUNT,
    DEFAULT_PRODUCT_COUNT,
    DEFAULT_RANDOM_SEED,
    ORDERS_FILE,
    PRODUCTS_FILE,
    RAW_DATA_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SyntheticDataGenerator:
    """Generate synthetic e-commerce data for customers, products, and orders.

    Args:
        customer_count: Number of customers to generate.
        product_count: Number of products to generate.
        order_count: Number of orders to generate.
        seed: Random seed for reproducible NumPy and Faker output.
        show_progress: Whether to display tqdm progress bars.
    """

    customer_count: int = 100_000
    product_count: int = 10_000
    order_count: int = 1_000_000
    seed: int = DEFAULT_RANDOM_SEED
    show_progress: bool = True
    fake: Faker = field(init=False)
    rng: Generator = field(init=False)

    def __post_init__(self) -> None:
        """Initialize Faker and NumPy random generators."""
        self.fake = Faker()
        Faker.seed(self.seed)
        self.rng = np.random.default_rng(self.seed)
        logger.debug(
            "Initialized generator with customers=%s products=%s orders=%s seed=%s",
            self.customer_count,
            self.product_count,
            self.order_count,
            self.seed,
        )

    def _progress(self, iterable: range, description: str) -> tqdm:
        """Wrap an iterable with tqdm when progress bars are enabled."""
        return tqdm(iterable, desc=description, disable=not self.show_progress)

    def generate_customers(self) -> pd.DataFrame:
        """Generate synthetic customer records.

        Customer ages are sampled from a normal distribution centered around 30
        and clipped to a realistic adult range.

        Returns:
            A pandas DataFrame with customer_id, name, email, age, city,
            country, and registration_date columns.
        """
        logger.info("Generating %s customers", self.customer_count)
        ages = self.rng.normal(loc=30, scale=10, size=self.customer_count)
        ages = np.clip(np.rint(ages), 18, 80).astype(int)

        registration_offsets = self.rng.integers(0, 1_095, size=self.customer_count)
        today = date.today()

        customers: list[dict[str, object]] = []
        for index in self._progress(range(self.customer_count), "Generating customers"):
            customers.append(
                {
                    "customer_id": index + 1,
                    "name": self.fake.name(),
                    "email": self.fake.unique.email(),
                    "age": int(ages[index]),
                    "city": self.fake.city(),
                    "country": self.fake.country(),
                    "registration_date": (today - timedelta(days=int(registration_offsets[index]))).isoformat(),
                }
            )

        dataframe = pd.DataFrame(customers)
        logger.debug("Generated customers shape: %s", dataframe.shape)
        return dataframe

    def generate_products(self) -> pd.DataFrame:
        """Generate synthetic product catalog records.

        Returns:
            A pandas DataFrame with product_id, name, category, price, stock,
            and rating columns.
        """
        logger.info("Generating %s products", self.product_count)
        categories = np.array(["Electronics", "Clothing", "Home", "Sports", "Books"])
        selected_categories = self.rng.choice(categories, size=self.product_count)
        prices = self.rng.uniform(10, 500, size=self.product_count).round(2)
        stocks = self.rng.integers(0, 1_001, size=self.product_count)
        ratings = self.rng.uniform(1, 5, size=self.product_count).round(1)

        products: list[dict[str, object]] = []
        for index in self._progress(range(self.product_count), "Generating products"):
            category = str(selected_categories[index])
            products.append(
                {
                    "product_id": index + 1,
                    "name": f"{self.fake.word().title()} {category}",
                    "category": category,
                    "price": float(prices[index]),
                    "stock": int(stocks[index]),
                    "rating": float(ratings[index]),
                }
            )

        dataframe = pd.DataFrame(products)
        logger.debug("Generated products shape: %s", dataframe.shape)
        return dataframe

    def generate_orders(self) -> pd.DataFrame:
        """Generate synthetic order records with Pareto-skewed customer demand.

        Customer IDs are sampled with probabilities derived from a Pareto
        distribution, which makes a small share of customers responsible for a
        large share of orders.

        Returns:
            A pandas DataFrame with order_id, customer_id, product_id, quantity,
            and order_date columns.
        """
        logger.info("Generating %s orders", self.order_count)
        customer_probabilities = self._pareto_customer_probabilities()
        customer_ids = self.rng.choice(
            np.arange(1, self.customer_count + 1),
            size=self.order_count,
            p=customer_probabilities,
        )
        product_ids = self.rng.integers(1, self.product_count + 1, size=self.order_count)
        quantities = self.rng.integers(1, 11, size=self.order_count)
        order_offsets = self.rng.integers(0, 730, size=self.order_count)
        start_date = date.today() - timedelta(days=730)

        order_dates: list[str] = []
        for offset in tqdm(order_offsets, desc="Generating order dates", disable=not self.show_progress):
            order_dates.append((start_date + timedelta(days=int(offset))).isoformat())

        dataframe = pd.DataFrame(
            {
                "order_id": np.arange(1, self.order_count + 1),
                "customer_id": customer_ids,
                "product_id": product_ids,
                "quantity": quantities,
                "order_date": order_dates,
            }
        )
        logger.debug("Generated orders shape: %s", dataframe.shape)
        return dataframe

    def _pareto_customer_probabilities(self) -> np.ndarray:
        """Build normalized Pareto weights for customer order assignment.

        Returns:
            A NumPy array of probabilities summing to 1.0.
        """
        weights = self.rng.pareto(a=1.16, size=self.customer_count) + 1
        weights = np.sort(weights)[::-1]
        probabilities = weights / weights.sum()
        logger.debug(
            "Top 20 percent customer order probability share: %.4f",
            probabilities[: max(1, self.customer_count // 5)].sum(),
        )
        return probabilities

    def generate_all(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate customers, products, and orders.

        Returns:
            A tuple of customers, products, and orders DataFrames.
        """
        customers = self.generate_customers()
        products = self.generate_products()
        orders = self.generate_orders()
        return customers, products, orders

    def save_dataframes(
        self,
        customers: pd.DataFrame,
        products: pd.DataFrame,
        orders: pd.DataFrame,
        output_dir: Path = RAW_DATA_DIR,
    ) -> None:
        """Save generated DataFrames as CSV files.

        Args:
            customers: Customer DataFrame to save.
            products: Product DataFrame to save.
            orders: Order DataFrame to save.
            output_dir: Directory where CSV files should be written.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        customers.to_csv(output_dir / "customers.csv", index=False)
        products.to_csv(output_dir / "products.csv", index=False)
        orders.to_csv(output_dir / "orders.csv", index=False)
        logger.info("Saved generated data to %s", output_dir)

    def generate_and_save(self, output_dir: Path = RAW_DATA_DIR) -> None:
        """Generate all datasets and save them as CSV files."""
        customers, products, orders = self.generate_all()
        self.save_dataframes(customers, products, orders, output_dir)


def generate_customers(count: int, seed: int = DEFAULT_RANDOM_SEED) -> pd.DataFrame:
    """Generate customer records using the class-based generator."""
    return SyntheticDataGenerator(customer_count=count, seed=seed, show_progress=False).generate_customers()


def generate_products(count: int, seed: int = DEFAULT_RANDOM_SEED) -> pd.DataFrame:
    """Generate product records using the class-based generator."""
    return SyntheticDataGenerator(product_count=count, seed=seed, show_progress=False).generate_products()


def generate_orders(
    count: int,
    customer_count: int,
    product_count: int,
    seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """Generate order records using the class-based generator."""
    return SyntheticDataGenerator(
        customer_count=customer_count,
        product_count=product_count,
        order_count=count,
        seed=seed,
    ).generate_orders()


def generate_all_data(
    customer_count: int = DEFAULT_CUSTOMER_COUNT,
    product_count: int = DEFAULT_PRODUCT_COUNT,
    order_count: int = DEFAULT_ORDER_COUNT,
    seed: int = DEFAULT_RANDOM_SEED,
) -> None:
    """Generate and save all raw e-commerce datasets."""
    generator = SyntheticDataGenerator(
        customer_count=customer_count,
        product_count=product_count,
        order_count=order_count,
        seed=seed,
    )
    generator.generate_and_save(RAW_DATA_DIR)


if __name__ == "__main__":
    SyntheticDataGenerator().generate_and_save()
