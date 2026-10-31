"""
Leakage-safe feature engineering.

A *snapshot* is a view of the business at a cutoff date:
  - features use ONLY transactions strictly before the cutoff
  - the label (churned) uses transactions in [cutoff, cutoff + CHURN_WINDOW_DAYS)
  - population = customers who bought in the ACTIVE_LOOKBACK_DAYS before the cutoff
"""
import numpy as np
import pandas as pd

from config import ACTIVE_LOOKBACK_DAYS, CHURN_WINDOW_DAYS

FEATURES = [
    "recency_days", "tenure_days", "frequency", "monetary", "avg_order_value",
    "avg_items_per_order", "avg_lines_per_order", "unique_products",
    "avg_gap_days", "std_gap_days", "recency_to_gap_ratio",
    "orders_30d", "orders_90d", "orders_180d", "revenue_30d", "revenue_90d", "revenue_180d",
    "orders_trend", "revenue_trend", "n_cancel_invoices", "cancel_rate", "is_uk",
]


def _window_stats(orders: pd.DataFrame, cutoff: pd.Timestamp, days: int, index) -> tuple[pd.Series, pd.Series]:
    w = orders[orders["order_date"] >= cutoff - pd.Timedelta(days=days)]
    g = w.groupby("customer_id")["order_value"]
    return g.size().reindex(index, fill_value=0), g.sum().reindex(index, fill_value=0.0)


def build_snapshot(df: pd.DataFrame, cutoff, with_label: bool = True) -> pd.DataFrame:
    """Return one row per active customer at `cutoff`: features (+ `churned` if with_label)."""
    cutoff = pd.Timestamp(cutoff)
    hist = df[df["invoice_date"] < cutoff]                       # <- the only data features may see
    sales = hist[~hist["is_cancellation"]]
    cancels = hist[hist["is_cancellation"]]

    recent = sales[sales["invoice_date"] >= cutoff - pd.Timedelta(days=ACTIVE_LOOKBACK_DAYS)]
    active_ids = recent["customer_id"].unique()
    sales = sales[sales["customer_id"].isin(active_ids)]
    if sales.empty:
        return pd.DataFrame(columns=["customer_id", *FEATURES, "cutoff"])

    # One row per invoice
    orders = (
        sales.groupby(["customer_id", "invoice"])
        .agg(order_date=("invoice_date", "min"), order_value=("revenue", "sum"),
             n_items=("quantity", "sum"), n_lines=("stock_code", "nunique"))
        .reset_index()
        .sort_values(["customer_id", "order_date"])
    )
    orders["gap_days"] = orders.groupby("customer_id")["order_date"].diff().dt.total_seconds() / 86400

    g = orders.groupby("customer_id")
    feat = pd.DataFrame({
        "frequency": g.size(),
        "monetary": g["order_value"].sum(),
        "avg_order_value": g["order_value"].mean(),
        "avg_items_per_order": g["n_items"].mean(),
        "avg_lines_per_order": g["n_lines"].mean(),
        "first_order": g["order_date"].min(),
        "last_order": g["order_date"].max(),
        "avg_gap_days": g["gap_days"].mean(),
        "std_gap_days": g["gap_days"].std(),
    })
    feat["recency_days"] = (cutoff - feat["last_order"]).dt.days
    feat["tenure_days"] = (cutoff - feat["first_order"]).dt.days
    feat["unique_products"] = sales.groupby("customer_id")["stock_code"].nunique()
    gap = feat["avg_gap_days"].where(feat["avg_gap_days"] > 0)
    feat["recency_to_gap_ratio"] = (feat["recency_days"] / gap).replace([np.inf, -np.inf], np.nan)

    idx = feat.index
    for days in (30, 90, 180):
        n, rev = _window_stats(orders, cutoff, days, idx)
        feat[f"orders_{days}d"], feat[f"revenue_{days}d"] = n, rev
    # Trend: last 90 days vs the 90 days before that
    prev = orders[(orders["order_date"] < cutoff - pd.Timedelta(days=90))
                  & (orders["order_date"] >= cutoff - pd.Timedelta(days=180))]
    pg = prev.groupby("customer_id")["order_value"]
    prev_n = pg.size().reindex(idx, fill_value=0)
    prev_rev = pg.sum().reindex(idx, fill_value=0.0)
    feat["orders_trend"] = feat["orders_90d"] - prev_n
    feat["revenue_trend"] = feat["revenue_90d"] - prev_rev

    n_cancel = cancels[cancels["customer_id"].isin(idx)].groupby("customer_id")["invoice"].nunique()
    feat["n_cancel_invoices"] = n_cancel.reindex(idx, fill_value=0)
    feat["cancel_rate"] = feat["n_cancel_invoices"] / (feat["frequency"] + feat["n_cancel_invoices"])

    last_country = sales.sort_values("invoice_date").groupby("customer_id")["country"].last()
    feat["is_uk"] = (last_country.reindex(idx) == "United Kingdom").astype(int)

    feat = feat.reset_index()
    feat["cutoff"] = cutoff

    if with_label:
        end = cutoff + pd.Timedelta(days=CHURN_WINDOW_DAYS)
        fut = df[(df["invoice_date"] >= cutoff) & (df["invoice_date"] < end) & (~df["is_cancellation"])]
        bought = set(fut["customer_id"].unique())
        feat["churned"] = (~feat["customer_id"].isin(bought)).astype(int)

    cols = ["customer_id", *FEATURES, "cutoff"] + (["churned"] if with_label else [])
    return feat[cols]


def build_dataset(df: pd.DataFrame, cutoffs) -> pd.DataFrame:
    """Stack several snapshots (used for training with multiple cutoffs)."""
    return pd.concat([build_snapshot(df, c) for c in cutoffs], ignore_index=True)
