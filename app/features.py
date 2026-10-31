"""Feature engineering shared by training (train_model.py) and the API.

Using ONE function for both avoids "training/serving skew": the model sees
exactly the same kind of features when it is trained and when it predicts.
"""
import numpy as np
import pandas as pd

TARGET = "Churn_Value"  # 1 = customer left, 0 = customer stayed

ADDON_COLS = [
    "Online_Security",
    "Online_Backup",
    "Device_Protection",
    "Tech_Support",
    "Streaming_TV",
    "Streaming_Movies",
]

# The 19 raw customer fields the API accepts
RAW_COLS = [
    "Gender",
    "Senior_Citizen",
    "Partner",
    "Dependents",
    "Tenure_Months",
    "Phone_Service",
    "Multiple_Lines",
    "Internet_Service",
    *ADDON_COLS,
    "Contract",
    "Paperless_Billing",
    "Payment_Method",
    "Monthly_Charges",
    "Total_Charges",
]

# The 4 features created in the notebook (Section 6)
ENGINEERED_COLS = ["ServiceCount", "TotalChargesPerTenure", "TenureGroup", "AutoPay"]

FEATURE_COLS = RAW_COLS + ENGINEERED_COLS

NUM_COLS = [
    "Tenure_Months",
    "Monthly_Charges",
    "Total_Charges",
    "TotalChargesPerTenure",
    "ServiceCount",
    "AutoPay",
]
CAT_COLS = [c for c in FEATURE_COLS if c not in NUM_COLS]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Take the raw columns and return the 23 model features (raw + engineered)."""
    out = df.copy()
    # number of add-on services the customer pays for (0-6)
    out["ServiceCount"] = (out[ADDON_COLS] == "Yes").sum(axis=1)
    # average spend per month of tenure; +1 avoids dividing by zero for brand-new customers
    out["TotalChargesPerTenure"] = out["Total_Charges"] / (out["Tenure_Months"] + 1)
    # lifecycle stage (upper edge is infinity so a tenure above 72 can never become NaN)
    out["TenureGroup"] = pd.cut(
        out["Tenure_Months"],
        bins=[-1, 12, 24, 48, np.inf],
        labels=["0-12 m", "13-24 m", "25-48 m", "49-72 m"],
    ).astype(str)
    # 1 if the customer pays automatically (bank transfer or credit card)
    out["AutoPay"] = out["Payment_Method"].str.contains("automatic").astype(int)
    return out[FEATURE_COLS]
