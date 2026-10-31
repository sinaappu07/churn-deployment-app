"""Tests that need no trained model: input validation and feature engineering."""
import pandas as pd
import pytest
from pydantic import ValidationError

from app.features import FEATURE_COLS, NUM_COLS, CAT_COLS, add_features
from app.schemas import HIGH_RISK_EXAMPLE, LOW_RISK_EXAMPLE, CustomerFeatures


def test_valid_examples_pass():
    CustomerFeatures(**HIGH_RISK_EXAMPLE)
    CustomerFeatures(**LOW_RISK_EXAMPLE)


@pytest.mark.parametrize(
    "field, value",
    [
        ("Tenure_Months", "12"),  # text instead of an integer
        ("Tenure_Months", -5),
        ("Tenure_Months", 500),
        ("Contract", "Weekly"),
        ("Monthly_Charges", -1),
        ("Gender", "Robot"),
    ],
)
def test_bad_values_are_rejected(field, value):
    with pytest.raises(ValidationError):
        CustomerFeatures(**{**HIGH_RISK_EXAMPLE, field: value})


def test_missing_and_unknown_fields_are_rejected():
    incomplete = {k: v for k, v in HIGH_RISK_EXAMPLE.items() if k != "Contract"}
    with pytest.raises(ValidationError):
        CustomerFeatures(**incomplete)
    with pytest.raises(ValidationError):
        CustomerFeatures(**{**HIGH_RISK_EXAMPLE, "Contrat": "Two year"})


def test_no_internet_fixes_addons_and_no_phone_fixes_lines():
    customer = CustomerFeatures(
        **{**HIGH_RISK_EXAMPLE, "Internet_Service": "No", "Phone_Service": "No", "Multiple_Lines": "Yes"}
    )
    assert customer.Online_Security == "No internet service"
    assert customer.Streaming_Movies == "No internet service"
    assert customer.Multiple_Lines == "No phone service"


def test_impossible_combination_is_rejected():
    with pytest.raises(ValidationError):
        CustomerFeatures(**{**HIGH_RISK_EXAMPLE, "Tech_Support": "No internet service"})


def test_total_charges_is_optional():
    data = {k: v for k, v in HIGH_RISK_EXAMPLE.items() if k != "Total_Charges"}
    assert CustomerFeatures(**data).Total_Charges is None


def test_feature_engineering_values():
    frame = pd.DataFrame([{**LOW_RISK_EXAMPLE}])
    out = add_features(frame)
    assert list(out.columns) == FEATURE_COLS
    row = out.iloc[0]
    assert row["ServiceCount"] == 4  # security, backup, device protection, tech support
    assert row["AutoPay"] == 1  # credit card (automatic)
    assert row["TenureGroup"] == "49-72 m"
    assert row["TotalChargesPerTenure"] == pytest.approx(3900.0 / 61)


def test_new_customer_has_no_division_by_zero():
    row = add_features(pd.DataFrame([{**HIGH_RISK_EXAMPLE, "Tenure_Months": 0, "Total_Charges": 0.0}])).iloc[0]
    assert row["TotalChargesPerTenure"] == 0
    assert row["TenureGroup"] == "0-12 m"


def test_column_groups_cover_all_features():
    assert sorted(NUM_COLS + CAT_COLS) == sorted(FEATURE_COLS)
