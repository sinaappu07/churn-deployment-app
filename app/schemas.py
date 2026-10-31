"""Request and response models. Pydantic checks every incoming value for us."""
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.features import ADDON_COLS

YesNo = Literal["Yes", "No"]
ServiceYesNo = Literal["Yes", "No", "No internet service"]

# Two ready-made customers, used for the API docs page, the tests and Postman.
HIGH_RISK_EXAMPLE = {
    "Gender": "Female",
    "Senior_Citizen": "No",
    "Partner": "No",
    "Dependents": "No",
    "Tenure_Months": 2,
    "Phone_Service": "Yes",
    "Multiple_Lines": "No",
    "Internet_Service": "Fiber optic",
    "Online_Security": "No",
    "Online_Backup": "No",
    "Device_Protection": "No",
    "Tech_Support": "No",
    "Streaming_TV": "No",
    "Streaming_Movies": "No",
    "Contract": "Month-to-month",
    "Paperless_Billing": "Yes",
    "Payment_Method": "Electronic check",
    "Monthly_Charges": 85.0,
    "Total_Charges": 170.0,
}

LOW_RISK_EXAMPLE = {
    "Gender": "Male",
    "Senior_Citizen": "No",
    "Partner": "Yes",
    "Dependents": "Yes",
    "Tenure_Months": 60,
    "Phone_Service": "Yes",
    "Multiple_Lines": "Yes",
    "Internet_Service": "DSL",
    "Online_Security": "Yes",
    "Online_Backup": "Yes",
    "Device_Protection": "Yes",
    "Tech_Support": "Yes",
    "Streaming_TV": "No",
    "Streaming_Movies": "No",
    "Contract": "Two year",
    "Paperless_Billing": "No",
    "Payment_Method": "Credit card (automatic)",
    "Monthly_Charges": 65.0,
    "Total_Charges": 3900.0,
}


class CustomerFeatures(BaseModel):
    """One customer, using the same field names and values as the training data."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": HIGH_RISK_EXAMPLE})

    Gender: Literal["Female", "Male"]
    Senior_Citizen: YesNo
    Partner: YesNo
    Dependents: YesNo
    # strict=True: a text value such as "12" is rejected, only a real integer is accepted
    Tenure_Months: int = Field(..., strict=True, ge=0, le=72, description="Months as a customer (0-72)")
    Phone_Service: YesNo
    Multiple_Lines: Literal["Yes", "No", "No phone service"]
    Internet_Service: Literal["DSL", "Fiber optic", "No"]
    Online_Security: ServiceYesNo
    Online_Backup: ServiceYesNo
    Device_Protection: ServiceYesNo
    Tech_Support: ServiceYesNo
    Streaming_TV: ServiceYesNo
    Streaming_Movies: ServiceYesNo
    Contract: Literal["Month-to-month", "One year", "Two year"]
    Paperless_Billing: YesNo
    Payment_Method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    Monthly_Charges: float = Field(..., ge=0, le=500, description="Monthly bill in dollars")
    Total_Charges: Optional[float] = Field(
        None,
        ge=0,
        le=20000,
        description="Total billed so far. Optional: if missing it is estimated as Monthly_Charges x Tenure_Months.",
    )

    @model_validator(mode="after")
    def check_service_consistency(self):
        """Fix or reject combinations that cannot exist in real customer data."""
        if self.Phone_Service == "No":
            self.Multiple_Lines = "No phone service"
        elif self.Multiple_Lines == "No phone service":
            raise ValueError("Multiple_Lines cannot be 'No phone service' when Phone_Service is 'Yes'")

        if self.Internet_Service == "No":
            for name in ADDON_COLS:
                setattr(self, name, "No internet service")
        else:
            wrong = [n for n in ADDON_COLS if getattr(self, n) == "No internet service"]
            if wrong:
                raise ValueError(
                    f"{', '.join(wrong)} cannot be 'No internet service' when Internet_Service is not 'No'"
                )
        return self


class PredictionResponse(BaseModel):
    churn_probability: float = Field(..., description="Probability (0-1) that the customer will leave")
    will_churn: bool = Field(..., description="True when the probability reaches the decision threshold")
    risk_level: Literal["Low", "Medium", "High"]
    decision_threshold: float
    recommendations: List[str]
    total_charges_used: float
    total_charges_estimated: bool
    model_name: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str] = None
    decision_threshold: Optional[float] = None
