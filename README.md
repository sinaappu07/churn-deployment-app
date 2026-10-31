# ConnectTel Churn Predictor: End-to-End Deployment

A trained machine-learning model turned into a live web application. A user enters a customer's details, and the app returns the probability that the customer will leave (churn), a risk level, and suggested retention actions.

This is the deployment (MLOps) companion to the notebook `Churn_Prediction_ConnectTel.ipynb` (EDA, three models, SHAP, business report).

```
 Browser ──► Streamlit UI ──HTTP──► FastAPI backend ──► XGBoost pipeline (model/)
             (frontend/)            (app/main.py)        preprocessing + model in ONE file
```

| Part | Technology | File |
|---|---|---|
| Model packaging | scikit-learn `Pipeline` + XGBoost, saved with joblib | `train_model.py`, `model/` |
| Backend API | FastAPI + Pydantic validation | `app/main.py`, `app/schemas.py` |
| Frontend | Streamlit | `frontend/streamlit_app.py` |
| Containerization | Docker | `Dockerfile`, `run.py` |
| Hosting | Render (any Docker host works) | see "Deploy" below |

## Repository structure

```
app/                 FastAPI backend (main.py = API code)
  features.py        feature engineering, shared by training and the API
  schemas.py         request/response models with input validation
  predictor.py       loads the model, makes predictions, builds recommendations
frontend/
  streamlit_app.py   web interface; talks to the API over HTTP
model/               churn_pipeline.joblib + metadata.json (created by train_model.py)
data/                Telco_customer_churn.xlsx (IBM sample dataset)
tests/               pytest tests
postman/             Postman collection for testing the API
train_model.py       trains the model and saves it to model/
run.py               container entry point (SERVICE=api or SERVICE=ui)
Dockerfile
requirements.txt
```

## The model

- Data: IBM Telco Customer Churn sample (7,043 customers).
- Pipeline: One-Hot Encoding + XGBoost (`n_estimators=200, learning_rate=0.05, max_depth=2`, the best settings found by GridSearchCV in the notebook), with `scale_pos_weight` for the class imbalance.
- 4 engineered features: `ServiceCount`, `TotalChargesPerTenure`, `TenureGroup`, `AutoPay`.
- **The preprocessing is saved inside the same file as the model**, so the API can never feed the model differently from how it was trained.
- Decision threshold: not the default 0.5. It is the highest threshold that still catches at least 75% of churners (recall >= 0.75), chosen with cross-validation on the training data. The value, test AUC, recall and precision are written to `model/metadata.json` when you train.

## Run it locally

Python 3.10 to 3.12 is recommended.

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# Mac / Linux:  source .venv/bin/activate

pip install -r requirements.txt
python train_model.py                       # creates model/churn_pipeline.joblib
uvicorn app.main:app --reload               # API at http://127.0.0.1:8000  (docs at /docs)
```

In a second terminal (same virtual environment):

```bash
streamlit run frontend/streamlit_app.py     # UI at http://localhost:8501
```

The UI finds the API through the `API_URL` environment variable (default `http://127.0.0.1:8000`).

Run the tests: `pip install -r requirements-dev.txt` then `pytest`.

## API

Interactive documentation: `GET /docs`.

### `GET /health`

```json
{"status": "ok", "model_loaded": true, "model_name": "XGBoost churn pipeline", "decision_threshold": 0.59}
```

### `POST /predict`

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "Gender": "Female", "Senior_Citizen": "No", "Partner": "No", "Dependents": "No",
  "Tenure_Months": 2, "Phone_Service": "Yes", "Multiple_Lines": "No",
  "Internet_Service": "Fiber optic", "Online_Security": "No", "Online_Backup": "No",
  "Device_Protection": "No", "Tech_Support": "No", "Streaming_TV": "No", "Streaming_Movies": "No",
  "Contract": "Month-to-month", "Paperless_Billing": "Yes", "Payment_Method": "Electronic check",
  "Monthly_Charges": 85.0, "Total_Charges": 170.0
}'
```

Response (fields):

| Field | Meaning |
|---|---|
| `churn_probability` | probability (0 to 1) that the customer leaves |
| `will_churn` | `true` when the probability reaches `decision_threshold` |
| `risk_level` | `High` (>= threshold), `Medium` (>= 0.35), otherwise `Low` |
| `recommendations` | rule-based retention actions (empty for Low risk) |
| `total_charges_used`, `total_charges_estimated` | `Total_Charges` is optional; if missing it is estimated as `Monthly_Charges x Tenure_Months` |

### Validation

Pydantic validates every request. Wrong input returns **HTTP 422** with a readable message, for example a text tenure (`"Tenure_Months": "twelve"`), a negative number, an unknown contract type, a missing or misspelt field. If `Internet_Service` is `No`, the add-on fields are set to `No internet service` automatically; if `Phone_Service` is `No`, `Multiple_Lines` becomes `No phone service`.

## Docker

```bash
docker build -t churn-app .
docker run --rm -p 8000:8000 churn-app                       # API on http://localhost:8000
docker run --rm -p 8501:8501 -e SERVICE=ui -e PORT=8501 \
           -e API_URL=http://host.docker.internal:8000 churn-app   # UI on http://localhost:8501
```

The image trains the model during the build (`RUN python train_model.py`), so the model always matches the library versions inside the image. One image, two services: `SERVICE=api` (default) starts FastAPI, `SERVICE=ui` starts Streamlit. The platform's `PORT` variable is respected.

## Deploy on Render (free tier)

Create two **Web Services** from this repository, both with *Language: Docker*:

| | API service | UI service |
|---|---|---|
| Name | `churn-api` | `churn-ui` |
| Environment variables | none | `SERVICE` = `ui`, `API_URL` = the public URL of the API service |
| Health check path | `/health` | leave empty |

Deploy the API first and copy its URL (for example `https://churn-api-xxxx.onrender.com`), then use it as `API_URL` for the UI service. Free services sleep after about 15 minutes without traffic and need about a minute to wake up on the next request.

## Limitations

- The probability shows a statistical association in historical data, not proof of cause. Test retention offers with an A/B experiment.
- Trained on one snapshot of California customers from a public sample dataset. Retrain regularly and monitor for drift before any real use.
- Only the data columns of the IBM Telco dataset are supported.
