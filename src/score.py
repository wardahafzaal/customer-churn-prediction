"""
Step 4a: batch-score all currently active customers.

    python src/score.py
Writes data/processed/scores.parquet (used by the API and the dashboard).
"""
import pandas as pd

from config import CLEAN_FILE, SCORES_FILE
from features import build_snapshot
from scoring import (expected_profit_if_contacted, load_artifact, predict_proba,
                     recommended_action, risk_tier, top_driver)


def main() -> None:
    df = pd.read_parquet(CLEAN_FILE)
    as_of = df["invoice_date"].max().normalize() + pd.Timedelta(days=1)
    snap = build_snapshot(df, as_of, with_label=False)
    art = load_artifact()

    X = snap[art["features"]]
    snap["churn_prob"] = predict_proba(art, X)
    snap["risk_tier"] = snap["churn_prob"].map(risk_tier)
    snap["top_driver"] = top_driver(art, X)
    hv = snap["monetary"].quantile(0.75)
    snap["recommended_action"] = [recommended_action(p, m, hv) for p, m in zip(snap["churn_prob"], snap["monetary"])]
    snap["expected_profit_if_contacted"] = expected_profit_if_contacted(snap["churn_prob"])
    snap["as_of"] = as_of

    SCORES_FILE.parent.mkdir(parents=True, exist_ok=True)
    snap.sort_values("churn_prob", ascending=False).to_parquet(SCORES_FILE, index=False)
    print(f"Scored {len(snap):,} customers as of {as_of.date()} -> {SCORES_FILE}")
    print(snap["risk_tier"].value_counts().to_string())


if __name__ == "__main__":
    main()
