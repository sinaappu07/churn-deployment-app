"""Train the churn model and save it into the model/ folder.

Run from the repository root:

    python train_model.py

It repeats the recipe from the Jupyter notebook: same cleaning, same 4 engineered
features, same 80/20 stratified split (seed 42), XGBoost with the best settings found
by GridSearchCV, and the decision threshold that keeps recall at 75% or more.

Outputs:
    model/churn_pipeline.joblib   preprocessing + XGBoost in ONE file
    model/metadata.json           threshold, metrics, feature list and library versions
"""
import argparse
import datetime as dt
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from app.features import CAT_COLS, FEATURE_COLS, NUM_COLS, RAW_COLS, TARGET, add_features

ROOT = Path(__file__).resolve().parent
RANDOM_STATE = 42
TARGET_RECALL = 0.75
# best settings found by GridSearchCV in the notebook (Section 9)
BEST_PARAMS = {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 2}


def load_dataset(path: Path) -> pd.DataFrame:
    """Read the IBM Telco Excel file and clean it exactly like the notebook did."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_excel(path)
    df.columns = df.columns.str.replace(" ", "_")
    missing = [c for c in RAW_COLS + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"The dataset is missing these columns: {missing}. Use Telco_customer_churn.xlsx (IBM).")
    # 11 brand-new customers have a blank Total Charges -> 0
    df["Total_Charges"] = pd.to_numeric(df["Total_Charges"], errors="coerce").fillna(0)
    return df


def make_pipeline(scale_pos_weight: float) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("num", "passthrough", NUM_COLS),
            ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False), CAT_COLS),
        ],
        verbose_feature_names_out=False,
    )
    model = XGBClassifier(
        **BEST_PARAMS,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,  # churners are the minority class
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    return Pipeline([("prep", preprocessor), ("model", model)])


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "Telco_customer_churn.xlsx")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "model")
    args = parser.parse_args(argv)

    df = load_dataset(args.data)
    X = add_features(df[RAW_COLS])
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    print(f"Training rows: {len(X_train):,} | test rows: {len(X_test):,} | churn rate: {y.mean():.1%}")

    # 1) threshold: out-of-fold predictions on the TRAINING data only (the test set stays untouched)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(make_pipeline(scale_pos_weight), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_train, oof)
    eligible = np.where(recall[:-1] >= TARGET_RECALL)[0]
    threshold = float(thresholds[eligible[-1]])  # highest threshold that still catches >= 75% of churners

    # 2) final model
    pipeline = make_pipeline(scale_pos_weight)
    pipeline.fit(X_train, y_train)

    # 3) honest evaluation on the held-out test set
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    metrics = {
        "test_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "test_recall": round(float(recall_score(y_test, pred)), 4),
        "test_precision": round(float(precision_score(y_test, pred)), 4),
    }
    print(f"Decision threshold: {threshold:.3f}")
    print(f"Test AUC: {metrics['test_auc']:.4f} | recall: {metrics['test_recall']:.0%} | precision: {metrics['test_precision']:.0%}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.output_dir / "churn_pipeline.joblib")
    metadata = {
        "model_name": "XGBoost churn pipeline",
        "decision_threshold": round(threshold, 4),
        "target_recall": TARGET_RECALL,
        **metrics,
        "xgboost_params": BEST_PARAMS,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "feature_columns": FEATURE_COLS,
        "library_versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "xgboost": getattr(xgboost, "__version__", "unknown"),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "trained_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved model and metadata to {args.output_dir}")


if __name__ == "__main__":
    main()
