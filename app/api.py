"""
Step 4b: FastAPI scoring service.

    uvicorn app.api:app --reload      (run from the project root)
Docs: http://127.0.0.1:8000/docs
"""
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from config import SCORES_FILE  # noqa: E402
from scoring import load_artifact, predict_proba, risk_tier  # noqa: E402

app = FastAPI(title="Customer Churn API", version="1.0")


@lru_cache(maxsize=1)
def artifact() -> dict:
    return load_artifact()


@lru_cache(maxsize=1)
def scores() -> pd.DataFrame:
    return pd.read_parquet(SCORES_FILE).set_index("customer_id")


class PredictRequest(BaseModel):
    customers: list[dict[str, float | None]] = Field(
        ..., description="List of feature dicts, e.g. {'recency_days': 45, 'frequency': 6, ...}. Missing features are treated as null.")


@app.get("/health")
def health() -> dict:
    try:
        art = artifact()
    except FileNotFoundError:
        raise HTTPException(503, "Model not found. Run `python src/train.py` first.")
    return {"status": "ok", "n_features": len(art["features"]), "break_even": art["break_even"]}


@app.get("/features")
def features() -> list[str]:
    return artifact()["features"]


@app.post("/predict")
def predict(req: PredictRequest) -> list[dict]:
    art = artifact()
    X = pd.DataFrame(req.customers).reindex(columns=art["features"])
    probs = predict_proba(art, X)
    return [{"churn_prob": round(float(p), 4), "risk_tier": risk_tier(p)} for p in probs]


@app.get("/score/{customer_id}")
def score(customer_id: int) -> dict:
    try:
        table = scores()
    except FileNotFoundError:
        raise HTTPException(503, "Scores not found. Run `python src/score.py` first.")
    if customer_id not in table.index:
        raise HTTPException(404, "Customer not found among currently active customers.")
    row = table.loc[customer_id]
    return {"customer_id": customer_id, "churn_prob": round(float(row["churn_prob"]), 4),
            "risk_tier": row["risk_tier"], "top_driver": row["top_driver"],
            "recommended_action": row["recommended_action"]}
