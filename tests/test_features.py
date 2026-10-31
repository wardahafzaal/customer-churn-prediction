import pandas as pd
import pytest

from features import FEATURES, build_snapshot


def make_df():
    rows = [
        # (invoice, date, customer, qty, price, cancel)
        ("1", "2011-01-10", 1, 2, 10.0, False),
        ("2", "2011-02-10", 1, 1, 20.0, False),
        ("3", "2011-03-01", 1, 1, 30.0, False),   # AFTER cutoff, inside churn window -> customer 1 NOT churned
        ("4", "2011-01-20", 2, 5, 4.0, False),
        ("C5", "2011-01-22", 2, -1, 4.0, True),
        ("6", "2010-01-01", 3, 1, 9.0, False),    # too old -> not in the active population
    ]
    df = pd.DataFrame(rows, columns=["invoice", "invoice_date", "customer_id", "quantity", "price", "is_cancellation"])
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["stock_code"], df["country"] = "10001", "United Kingdom"
    df["revenue"] = df["quantity"] * df["price"]
    return df


CUTOFF = "2011-02-20"


def test_population_is_active_customers_only():
    snap = build_snapshot(make_df(), CUTOFF)
    assert set(snap["customer_id"]) == {1, 2}


def test_labels():
    snap = build_snapshot(make_df(), CUTOFF).set_index("customer_id")
    assert snap.loc[1, "churned"] == 0      # bought on 2011-03-01, inside the 90-day window
    assert snap.loc[2, "churned"] == 1      # never bought again


def test_no_future_leakage_in_features():
    df = make_df()
    full = build_snapshot(df, CUTOFF, with_label=False)
    truncated = build_snapshot(df[df["invoice_date"] < CUTOFF], CUTOFF, with_label=False)
    pd.testing.assert_frame_equal(full.reset_index(drop=True), truncated.reset_index(drop=True))


def test_feature_values():
    snap = build_snapshot(make_df(), CUTOFF).set_index("customer_id")
    c1 = snap.loc[1]
    assert c1["frequency"] == 2
    assert c1["monetary"] == pytest.approx(40.0)       # 2*10 + 1*20
    assert c1["recency_days"] == 10                    # 2011-02-10 -> 2011-02-20
    assert c1["avg_gap_days"] == pytest.approx(31.0)
    assert snap.loc[2, "n_cancel_invoices"] == 1
    assert snap.loc[2, "cancel_rate"] == pytest.approx(0.5)


def test_all_feature_columns_present():
    snap = build_snapshot(make_df(), CUTOFF)
    assert all(f in snap.columns for f in FEATURES)
