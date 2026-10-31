"""
Step 4c: Streamlit dashboard of at-risk customers.

    streamlit run app/dashboard.py      (run from the project root)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from config import OFFER_COST, RISK_HIGH, SAVE_RATE, SCORES_FILE, VALUE_IF_SAVED  # noqa: E402

st.set_page_config(page_title="Churn dashboard", layout="wide")
st.title("Customer churn risk")

if not Path(SCORES_FILE).exists():
    st.error("No scores found. Run `python src/train.py` then `python src/score.py`.")
    st.stop()

df = pd.read_parquet(SCORES_FILE)
st.caption(f"Scores as of {pd.Timestamp(df['as_of'].iloc[0]).date()} | "
           f"churn = no purchase in the next 90 days | {len(df):,} active customers")

with st.sidebar:
    st.header("Campaign economics")
    cost = st.number_input("Offer cost per customer (GBP)", 0.0, 500.0, float(OFFER_COST), 1.0)
    value = st.number_input("Value of a saved customer (GBP)", 0.0, 5000.0, float(VALUE_IF_SAVED), 10.0)
    save_rate = st.slider("Offer success rate", 0.01, 1.0, float(SAVE_RATE), 0.01)
    st.header("Filters")
    countries = st.multiselect("Segment", ["UK", "Non-UK"], default=["UK", "Non-UK"])
    tiers = st.multiselect("Risk tier", ["high", "medium", "low"], default=["high", "medium"])

df["exp_profit"] = df["churn_prob"] * save_rate * value - cost
break_even = cost / (save_rate * value) if save_rate * value > 0 else 1.0
mask = df["risk_tier"].isin(tiers)
mask &= ((df["is_uk"] == 1) & ("UK" in countries)) | ((df["is_uk"] == 0) & ("Non-UK" in countries))
view = df[mask]
target = df[df["churn_prob"] >= break_even]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Active customers", f"{len(df):,}")
c2.metric("High risk", f"{(df['churn_prob'] >= RISK_HIGH).sum():,}")
c3.metric("Break-even probability", f"{break_even:.0%}")
c4.metric("Expected profit (contact above break-even)", f"£{target['exp_profit'].sum():,.0f}")

st.subheader("Distribution of churn probability")
hist = pd.cut(df["churn_prob"], bins=[i / 10 for i in range(11)], include_lowest=True).value_counts().sort_index()
st.bar_chart(pd.Series(hist.values, index=[f"{iv.left:.1f}-{iv.right:.1f}" for iv in hist.index]))

st.subheader(f"Customers to act on ({len(view):,})")
cols = ["customer_id", "churn_prob", "risk_tier", "top_driver", "recommended_action", "recency_days",
        "frequency", "monetary", "exp_profit"]
st.dataframe(view[cols].sort_values("churn_prob", ascending=False).head(500), width="stretch", hide_index=True)
st.download_button("Download list (CSV)", view[cols].to_csv(index=False), "at_risk_customers.csv", "text/csv")
