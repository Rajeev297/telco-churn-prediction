from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class PredictionRequest(BaseModel):
    customers: List[CustomerData] = Field(..., max_length=10000, description="List of customers to predict")


class SinglePrediction(BaseModel):
    churn_probability: float
    churn_prediction: str
    risk_level: str


class PredictionResponse(BaseModel):
    predictions: List[SinglePrediction]
    summary: Dict[str, Any]


class BulkPredictionResponse(BaseModel):
    total_customers: int
    predictions: List[SinglePrediction]
    summary: Dict[str, Any]
    download_url: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    threshold: float
