"""FastAPI backend.

    GET  /health   -> is the service alive and is the model loaded?
    POST /predict  -> JSON customer in, churn probability and advice out
    GET  /docs     -> automatic interactive API documentation (Swagger UI)
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from app.predictor import ChurnPredictor
from app.schemas import CustomerFeatures, HealthResponse, PredictionResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("churn-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The model is loaded ONCE when the server starts, not on every request.
    try:
        app.state.predictor = ChurnPredictor.load()
        logger.info("Model loaded (decision threshold %.3f)", app.state.predictor.threshold)
    except Exception:
        logger.exception("Could not load the model. Run `python train_model.py` first.")
        app.state.predictor = None
    yield


app = FastAPI(
    title="ConnectTel Churn Prediction API",
    description="Send customer details, get back the probability that the customer will churn.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/", include_in_schema=False)
def root():
    return {"service": "ConnectTel Churn Prediction API", "docs": "/docs", "health": "/health", "predict": "POST /predict"}


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        return HealthResponse(status="degraded", model_loaded=False)
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_name=predictor.model_name,
        decision_threshold=predictor.threshold,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures, request: Request):
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Check the server logs.")
    try:
        return predictor.predict(customer)
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed. Check the server logs.")
