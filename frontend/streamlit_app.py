"""Streamlit frontend for the ConnectTel churn predictor.

This page never loads the model. It only sends HTTP requests to the FastAPI backend
(set its address with the API_URL environment variable) and shows the answer.
"""
import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
# A free cloud API goes to sleep when idle and can need about a minute to wake up
REQUEST_TIMEOUT = 90

st.set_page_config(page_title="ConnectTel Churn Predictor", page_icon="📞", layout="wide")

YES_NO = ["No", "Yes"]
GENDERS = ["Female", "Male"]
INTERNET = ["Fiber optic", "DSL", "No"]
CONTRACTS = ["Month-to-month", "One year", "Two year"]
PAYMENTS = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
ADDONS = [
    ("Online_Security", "Online security"),
    ("Online_Backup", "Online backup"),
    ("Device_Protection", "Device protection"),
    ("Tech_Support", "Tech support"),
    ("Streaming_TV", "Streaming TV"),
    ("Streaming_Movies", "Streaming movies"),
]

DEFAULT_CUSTOMER = {
    "Gender": "Female", "Senior_Citizen": "No", "Partner": "No", "Dependents": "No",
    "Tenure_Months": 12, "Phone_Service": "Yes", "Multiple_Lines": "No",
    "Internet_Service": "Fiber optic",
    "Online_Security": "No", "Online_Backup": "No", "Device_Protection": "No",
    "Tech_Support": "No", "Streaming_TV": "No", "Streaming_Movies": "No",
    "Contract": "Month-to-month", "Paperless_Billing": "Yes", "Payment_Method": "Electronic check",
    "Monthly_Charges": 75.0, "Total_Charges": 0.0,
}
HIGH_RISK = {
    **DEFAULT_CUSTOMER,
    "Tenure_Months": 2, "Monthly_Charges": 85.0, "Total_Charges": 170.0,
}
LOW_RISK = {
    "Gender": "Male", "Senior_Citizen": "No", "Partner": "Yes", "Dependents": "Yes",
    "Tenure_Months": 60, "Phone_Service": "Yes", "Multiple_Lines": "Yes",
    "Internet_Service": "DSL",
    "Online_Security": "Yes", "Online_Backup": "Yes", "Device_Protection": "Yes",
    "Tech_Support": "Yes", "Streaming_TV": "No", "Streaming_Movies": "No",
    "Contract": "Two year", "Paperless_Billing": "No", "Payment_Method": "Credit card (automatic)",
    "Monthly_Charges": 65.0, "Total_Charges": 3900.0,
}

# first visit: fill every widget with a default value
if "initialized" not in st.session_state:
    for _key, _value in DEFAULT_CUSTOMER.items():
        st.session_state[_key] = _value
    st.session_state["result"] = None
    st.session_state["initialized"] = True


def load_example(example):
    for key, value in example.items():
        st.session_state[key] = value
    st.session_state["result"] = None


@st.cache_data(ttl=20, show_spinner=False)
def api_is_online(url):
    """True = online with model, False = online but model missing, None = not reachable."""
    try:
        response = requests.get(f"{url}/health", timeout=5)
        return bool(response.status_code == 200 and response.json().get("model_loaded"))
    except Exception:
        return None


def call_api(payload):
    """Send the customer to POST /predict and turn every outcome into a simple dict."""
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.ConnectionError:
        return {"kind": "error", "message": f"Cannot reach the API at {API_URL}. If it is hosted on a free plan it may be waking up: wait a minute and try again."}
    except requests.exceptions.Timeout:
        return {"kind": "error", "message": "The API took too long to answer. Please try again in a minute."}

    if response.status_code == 200:
        return {"kind": "ok", "data": response.json()}

    if response.status_code == 422:
        messages = []
        try:
            detail = response.json().get("detail", [])
        except ValueError:
            detail = []
        if isinstance(detail, list):
            for item in detail:
                where = ".".join(str(part) for part in item.get("loc", [])[1:]) or "input"
                messages.append(f"{where}: {item.get('msg', 'invalid value')}")
        else:
            messages.append(str(detail))
        return {"kind": "invalid", "messages": messages or ["The API rejected the input."]}

    return {"kind": "error", "message": f"The API returned HTTP {response.status_code}: {response.text[:200]}"}


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Backend API")
    st.code(API_URL, language=None)
    status = api_is_online(API_URL)
    if status:
        st.success("API online, model loaded")
    elif status is False:
        st.warning("API online but the model is not loaded")
    else:
        st.error("API not reachable (it may be waking up)")
    st.markdown(f"[Open API docs]({API_URL}/docs)")
    st.divider()
    st.subheader("Try an example")
    st.button("High-risk customer", on_click=load_example, args=(HIGH_RISK,))
    st.button("Low-risk customer", on_click=load_example, args=(LOW_RISK,))

# ------------------------------------------------------------------ page
st.title("📞 ConnectTel Churn Predictor")
st.write("Enter a customer's details to estimate the chance that they will leave, and get suggested retention actions.")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Customer")
    gender = st.selectbox("Gender", GENDERS, key="Gender")
    senior = st.selectbox("Senior citizen", YES_NO, key="Senior_Citizen")
    partner = st.selectbox("Has a partner", YES_NO, key="Partner")
    dependents = st.selectbox("Has dependents", YES_NO, key="Dependents")
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, step=1, key="Tenure_Months")

with col2:
    st.subheader("Services")
    phone = st.selectbox("Phone service", YES_NO, key="Phone_Service")
    if phone == "Yes":
        lines = st.selectbox("Multiple lines", YES_NO, key="Multiple_Lines")
    else:
        lines = "No phone service"
        st.caption("Multiple lines: not applicable without phone service")
    internet = st.selectbox("Internet service", INTERNET, key="Internet_Service")
    addon_values = {}
    if internet == "No":
        st.caption("Add-on services: not applicable without internet")
        for field, _label in ADDONS:
            addon_values[field] = "No internet service"
    else:
        for field, label in ADDONS:
            addon_values[field] = st.selectbox(label, YES_NO, key=field)

with col3:
    st.subheader("Account and billing")
    contract = st.selectbox("Contract", CONTRACTS, key="Contract")
    paperless = st.selectbox("Paperless billing", YES_NO, key="Paperless_Billing")
    payment = st.selectbox("Payment method", PAYMENTS, key="Payment_Method")
    monthly = st.number_input("Monthly charges ($)", min_value=0.0, max_value=500.0, step=0.5, key="Monthly_Charges")
    total = st.number_input("Total charges so far ($), 0 = estimate", min_value=0.0, max_value=20000.0, step=10.0, key="Total_Charges")

if st.button("Predict churn risk", type="primary"):
    payload = {
        "Gender": gender,
        "Senior_Citizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "Tenure_Months": int(tenure),
        "Phone_Service": phone,
        "Multiple_Lines": lines,
        "Internet_Service": internet,
        **addon_values,
        "Contract": contract,
        "Paperless_Billing": paperless,
        "Payment_Method": payment,
        "Monthly_Charges": float(monthly),
    }
    if total > 0:
        payload["Total_Charges"] = float(total)
    with st.spinner("Asking the API..."):
        st.session_state["result"] = call_api(payload)

# ------------------------------------------------------------------ result
result = st.session_state.get("result")
if result:
    st.divider()
    if result["kind"] == "ok":
        data = result["data"]
        probability = float(data["churn_probability"])
        level = data["risk_level"]

        m1, m2, m3 = st.columns(3)
        m1.metric("Churn probability", f"{probability:.1%}")
        m2.metric("Risk level", level)
        m3.metric("Decision threshold", f"{data['decision_threshold']:.2f}")
        st.progress(min(max(probability, 0.0), 1.0))

        if level == "High":
            st.error("High risk: this customer is likely to leave. Act now.")
        elif level == "Medium":
            st.warning("Medium risk: keep an eye on this customer.")
        else:
            st.success("Low risk: this customer looks stable.")

        if data["recommendations"]:
            st.markdown("**Suggested retention actions**")
            for action in data["recommendations"]:
                st.markdown(f"- {action}")
        if data.get("total_charges_estimated"):
            st.caption(f"Total charges were estimated as ${data['total_charges_used']:,.2f} (monthly charges x tenure).")
        with st.expander("Raw API response"):
            st.json(data)

    elif result["kind"] == "invalid":
        st.error("Invalid input. Please fix the following and try again:")
        for message in result["messages"]:
            st.markdown(f"- {message}")
    else:
        st.error(result["message"])

st.caption(
    "Model: XGBoost trained on the IBM Telco sample dataset. The probability shows association in historical data, "
    "not proof of cause. Validate any retention offer with an A/B test."
)
