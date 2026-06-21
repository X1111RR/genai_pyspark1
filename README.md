# E-Commerce Data Pipeline

This project generates synthetic e-commerce data and analyzes it with PySpark. It is useful for testing customer, product, and order workflows without using real production data.

## Project Structure

```text
genai_pyspark1/
|-- data/
|   |-- raw/
|   |   `-- .gitkeep
|   `-- processed/
|       `-- .gitkeep
|-- notebooks/
|   `-- .gitkeep
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- data_generator.py
|   `-- spark_analytics.py
|-- tests/
|   `-- test_data_generator.py
|-- .gitignore
|-- LICENSE
|-- README.md
`-- requirements.txt
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

PySpark requires Java to be installed and available on your `PATH`.

## Generate Raw Data

```bash
python -m src.data_generator
```

This creates:

- `data/raw/customers.csv`
- `data/raw/products.csv`
- `data/raw/orders.csv`

## Run PySpark Analytics

```bash
python -m src.spark_analytics
```

This writes analyzed CSV outputs under `data/processed/`:

- `revenue_by_category/`
- `top_customers/`
- `monthly_sales/`
- `order_status_summary/`

## Run Tests

```bash
pytest
```

## Business Insights Produced

- Revenue by product category
- Highest-value customers
- Monthly completed sales trends
- Order status distribution
