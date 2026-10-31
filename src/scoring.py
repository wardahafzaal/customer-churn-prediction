"""Shared inference helpers used by score.py, the API and the dashboard."""
import joblib
import numpy as np
import pandas as pd

from config import MODEL_FILE, RISK_HIGH, RISK_MEDIUM, OFFER_COST, SAVE_RATE, VALUE_IF_SAVED


def load_artifact(path=MODEL_FILE) -> dict:
    return joblib.load(path)


def predict_proba(artifact: dict, X: pd.DataFrame) -> np.ndarray:
    """Calibrated churn probability."""
    X = X.reindex(columns=artifact["features"])
    raw = artifact["model"].predict_proba(X)[:, 1]
    return np.clip(artifact["calibrator"].predict(raw), 0.0, 1.0)


def top_driver(artifact: dict, X: pd.DataFrame) -> pd.Series:
    """Feature with the largest push toward churn for each customer (LightGBM native SHAP)."""
    X = X.reindex(columns=artifact["features"])
    contrib = artifact["model"].booster_.predict(X, pred_contrib=True)[:, :-1]
    names = np.array(artifact["features"])
    return pd.Series(names[contrib.argmax(axis=1)], index=X.index)


def risk_tier(p: float) -> str:
    return "high" if p >= RISK_HIGH else "medium" if p >= RISK_MEDIUM else "low"


def recommended_action(p: float, monetary: float, high_value_cutoff: float) -> str:
    tier = risk_tier(p)
    if tier == "high" and monetary >= high_value_cutoff:
        return "Personal outreach (account manager call)"
    if tier == "high":
        return "Win-back discount email"
    if tier == "medium":
        return "Reminder email with new arrivals"
    return "No action"


def expected_profit_if_contacted(p, cost=OFFER_COST, save_rate=SAVE_RATE, value=VALUE_IF_SAVED):
    return p * save_rate * value - cost
