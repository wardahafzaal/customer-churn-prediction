import pytest
from fastapi.testclient import TestClient

from config import MODEL_FILE, SCORES_FILE

pytestmark = pytest.mark.skipif(not (MODEL_FILE.exists() and SCORES_FILE.exists()),
                                reason="run train.py and score.py first")


@pytest.fixture(scope="module")
def client():
    from app.api import app
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_returns_probabilities(client):
    r = client.post("/predict", json={"customers": [
        {"recency_days": 5, "frequency": 20, "monetary": 5000, "is_uk": 1},
        {"recency_days": 170, "frequency": 1, "monetary": 50, "is_uk": 1}]})
    assert r.status_code == 200
    out = r.json()
    assert all(0 <= o["churn_prob"] <= 1 for o in out)
    assert out[1]["churn_prob"] > out[0]["churn_prob"]    # long-silent one-time buyer is riskier


def test_unknown_customer_404(client):
    assert client.get("/score/1").status_code == 404
