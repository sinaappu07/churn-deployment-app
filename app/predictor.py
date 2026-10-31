"""Loads the saved model once and turns one customer into a prediction."""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import pandas as pd
import sklearn

from app.features import add_features
from app.schemas import CustomerFeatures

logger = logging.getLogger("churn-api")

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_FILE = "churn_pipeline.joblib"
METADATA_FILE = "metadata.json"

DEFAULT_THRESHOLD = 0.59  # used only if metadata.json is missing
MEDIUM_RISK_FLOOR = 0.35  # between this value and the threshold = "Medium"
MODEL_NAME = "XGBoost churn pipeline"


def risk_level(probability: float, threshold: float) -> str:
    if probability >= threshold:
        return "High"
    if probability >= MEDIUM_RISK_FLOOR:
        return "Medium"
    return "Low"


def build_recommendations(data: Dict[str, Any], level: str) -> List[str]:
    """Rule-based retention ideas taken from the Business Insight Report in the notebook."""
    if level == "Low":
        return []
    actions = []
    if data["Contract"] == "Month-to-month":
        actions.append("Offer a 1-year contract discount or a free add-on in exchange for a 12-month commitment.")
    if data["Payment_Method"] == "Electronic check":
        actions.append("Offer a small one-time credit for switching to automatic payment.")
    if data["Internet_Service"] != "No" and (data["Tech_Support"] == "No" or data["Online_Security"] == "No"):
        actions.append("Offer a free 3-month trial of tech support or online security.")
    if data["Tenure_Months"] <= 12:
        actions.append("Add the customer to the first-year retention programme (proactive onboarding check-ins).")
    if not actions:
        actions.append("Schedule a proactive check-in call and review the customer's plan.")
    return actions


class ChurnPredictor:
    def __init__(self, pipeline: Any, metadata: Dict[str, Any]):
        self.pipeline = pipeline
        self.metadata = metadata
        self.threshold = float(metadata.get("decision_threshold", DEFAULT_THRESHOLD))
        self.model_name = metadata.get("model_name", MODEL_NAME)

    @classmethod
    def load(cls, model_dir: Optional[Union[str, Path]] = None) -> "ChurnPredictor":
        folder = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        model_path = folder / MODEL_FILE
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} not found. Create it first with: python train_model.py")

        pipeline = joblib.load(model_path)

        metadata: Dict[str, Any] = {}
        meta_path = folder / METADATA_FILE
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            trained_with = metadata.get("library_versions", {}).get("scikit-learn")
            if trained_with and trained_with != sklearn.__version__:
                logger.warning(
                    "Model was trained with scikit-learn %s but %s is installed. "
                    "Re-run train_model.py if predictions fail.",
                    trained_with,
                    sklearn.__version__,
                )
        return cls(pipeline, metadata)

    def predict(self, customer: CustomerFeatures) -> Dict[str, Any]:
        data = customer.model_dump()

        estimated = data["Total_Charges"] is None
        if estimated:
            data["Total_Charges"] = round(data["Monthly_Charges"] * data["Tenure_Months"], 2)

        features = add_features(pd.DataFrame([data]))
        probability = float(self.pipeline.predict_proba(features)[0, 1])
        level = risk_level(probability, self.threshold)

        return {
            "churn_probability": round(probability, 4),
            "will_churn": probability >= self.threshold,
            "risk_level": level,
            "decision_threshold": round(self.threshold, 4),
            "recommendations": build_recommendations(data, level),
            "total_charges_used": data["Total_Charges"],
            "total_charges_estimated": estimated,
            "model_name": self.model_name,
        }
