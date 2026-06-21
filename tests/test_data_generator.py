"""Tests for synthetic data generation."""

import pandas as pd

from src.data_generator import SyntheticDataGenerator, generate_customers, generate_orders, generate_products


def test_generate_customers_count_and_columns() -> None:
    """Customers should contain the requested number of rows and key columns."""
    customers = generate_customers(5)

    assert len(customers) == 5
    assert {"customer_id", "name", "email", "age", "city", "country", "registration_date"}.issubset(
        customers.columns
    )
    assert customers["age"].between(18, 80).all()


def test_generate_products_count_and_values() -> None:
    """Products should contain valid prices, ratings, and categories."""
    products = generate_products(5)

    assert len(products) == 5
    assert (products["price"] >= 10).all()
    assert (products["price"] <= 500).all()
    assert products["rating"].between(1, 5).all()


def test_generate_orders_links_to_valid_ids() -> None:
    """Orders should reference valid customer and product ID ranges."""
    orders = generate_orders(count=20, customer_count=5, product_count=3)

    assert len(orders) == 20
    assert orders["customer_id"].between(1, 5).all()
    assert orders["product_id"].between(1, 3).all()
    assert orders["quantity"].between(1, 10).all()
    assert isinstance(orders, pd.DataFrame)


def test_class_generator_returns_all_dataframes() -> None:
    """The class API should generate all three e-commerce DataFrames."""
    generator = SyntheticDataGenerator(
        customer_count=5,
        product_count=4,
        order_count=20,
        show_progress=False,
    )
    customers, products, orders = generator.generate_all()

    assert len(customers) == 5
    assert len(products) == 4
    assert len(orders) == 20
