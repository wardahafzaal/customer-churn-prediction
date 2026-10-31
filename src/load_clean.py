"""
Step 1: load, clean and explore the transactions.

Usage:
    python src/load_clean.py                          # real data: data/raw/online_retail_II.xlsx
    python src/load_clean.py --input data/raw/synthetic_retail.csv
"""
import argparse
from pathlib import Path

import pandas as pd

from config import CLEAN_FILE, RAW_FILE, TEST_CUTOFF

COLUMN_MAP = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "price",
    "Customer ID": "customer_id",
    "Country": "country",
}


def load_raw(path: Path = RAW_FILE) -> pd.DataFrame:
    """Read the raw file. Excel files have two sheets (2009-10 and 2010-11) that we stack."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download online_retail_II.xlsx from "
            "https://archive.ics.uci.edu/dataset/502/online+retail+ii into data/raw/ "
            "or generate test data with: python src/make_synthetic_data.py"
        )
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, dtype={"StockCode": str, "Invoice": str})
    else:
        df = pd.concat(pd.read_excel(path, sheet_name=None).values(), ignore_index=True)
    return df.rename(columns=COLUMN_MAP)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning steps. Each prints how many rows it removes so you can document them."""
    n0 = len(df)
    df = df.copy()
    df["invoice"] = df["invoice"].astype(str)
    df["stock_code"] = df["stock_code"].astype(str)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["is_cancellation"] = df["invoice"].str.startswith("C")

    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df):,} duplicate rows")

    before = len(df)
    df = df[df["customer_id"].notna()].copy()
    df["customer_id"] = df["customer_id"].astype(int)
    print(f"Dropped {before - len(df):,} rows with missing customer_id (guest checkouts)")

    before = len(df)
    df = df[df["stock_code"].str.match(r"^\d{5}")].copy()
    print(f"Dropped {before - len(df):,} non-product rows (POST, M, BANK CHARGES, ...)")

    before = len(df)
    df = df[df["price"] > 0]
    df = df[(df["quantity"] > 0) | df["is_cancellation"]].copy()
    print(f"Dropped {before - len(df):,} rows with bad price/quantity")

    df["revenue"] = df["quantity"] * df["price"]
    print(f"Kept {len(df):,} of {n0:,} rows ({len(df) / n0:.1%})")
    return df.reset_index(drop=True)


def purchase_summary(df: pd.DataFrame) -> None:
    sales = df[~df["is_cancellation"]]
    orders = sales.groupby("customer_id")["invoice"].nunique()
    print("\n=== Customer purchase summary ===")
    print(f"Customers: {len(orders):,}")
    print(f"Customers with 2+ orders: {(orders > 1).mean():.1%}")
    print(orders.describe().round(1).to_string())
    print(f"Date range: {sales['invoice_date'].min().date()} -> {sales['invoice_date'].max().date()}")


def churn_label_preview(df: pd.DataFrame) -> None:
    from features import build_snapshot  # local import avoids circularity

    snap = build_snapshot(df, TEST_CUTOFF)
    print("\n=== Churn label preview (test snapshot) ===")
    print(f"Cutoff: {TEST_CUTOFF} | active customers: {len(snap):,} | churn rate: {snap['churned'].mean():.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(RAW_FILE))
    args = parser.parse_args()

    raw = load_raw(Path(args.input))
    print(f"Raw shape: {raw.shape}")
    clean_df = clean(raw)
    CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_parquet(CLEAN_FILE, index=False)
    print(f"Saved -> {CLEAN_FILE}")
    purchase_summary(clean_df)
    churn_label_preview(clean_df)
