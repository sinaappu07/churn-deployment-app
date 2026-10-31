"""API tests. They run only when fastapi and httpx are installed and model/ has been trained."""
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.schemas import HIGH_RISK_EXAMPLE, LOW_RISK_EXAMPLE  # noqa: E402

MODEL_FILE = Path(__file__).resolve().parent.parent / "model" / "churn_pipeline.joblib"
pytestmark = pytest.mark.skipif(not MODEL_FILE.exists(), reason="run `python train_model.py` first")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:  # the "with" block runs the startup that loads the model
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_high_risk_is_higher_than_low_risk(client):
    high = client.post("/predict", json=HIGH_RISK_EXAMPLE)
    low = client.post("/predict", json=LOW_RISK_EXAMPLE)
    assert high.status_code == 200 and low.status_code == 200
    assert high.json()["churn_probability"] > low.json()["churn_probability"]
    assert 0.0 <= low.json()["churn_probability"] <= 1.0


def test_invalid_input_returns_422(client):
    response = client.post("/predict", json={**HIGH_RISK_EXAMPLE, "Tenure_Months": -5})
    assert response.status_code == 422
