"""
FastAPI app for serving the Pima Indians Diabetes prediction model.

Run with:
    uvicorn main:app --reload

Test with:
    http://127.0.0.1:8000/docs   <-- interactive Swagger UI (built into FastAPI)
"""

import logging
import time
from contextlib import asynccontextmanager

import joblib
import numpy as np
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

# ----------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("diabetes_api")

# ----------------------------------------------------------------------
# Load model + scaler once at startup (not on every request)
# ----------------------------------------------------------------------
MODEL_PATH = "diabetes_model.pkl"
SCALER_PATH = "scaler.pkl"

model = None
scaler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler
    logger.info("Loading model and scaler...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logger.info("Model and scaler loaded successfully.")
    yield
    logger.info("Shutting down API.")


app = FastAPI(
    title="Diabetes Prediction API",
    description="Predicts diabetes risk from the Pima Indians dataset features.",
    version="1.0.0",
    lifespan=lifespan,
)

# ----------------------------------------------------------------------
# Logging middleware - logs every request with method, path, status, time
# ----------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000

    logger.info(
        f"{request.method} {request.url.path} "
        f"-> {response.status_code} ({duration_ms:.1f}ms)"
    )
    return response


# ----------------------------------------------------------------------
# Pydantic schema for input validation
# ----------------------------------------------------------------------
class DiabetesInput(BaseModel):
    Pregnancies: int = Field(..., ge=0, le=20, description="Number of pregnancies")
    Glucose: float = Field(..., ge=0, le=300, description="Plasma glucose concentration")
    BloodPressure: float = Field(..., ge=0, le=200, description="Diastolic blood pressure (mm Hg)")
    SkinThickness: float = Field(..., ge=0, le=100, description="Triceps skin fold thickness (mm)")
    Insulin: float = Field(..., ge=0, le=900, description="2-Hour serum insulin (mu U/ml)")
    BMI: float = Field(..., ge=0, le=70, description="Body mass index")
    DiabetesPedigreeFunction: float = Field(..., ge=0, le=3, description="Diabetes pedigree function score")
    Age: int = Field(..., ge=1, le=120, description="Age in years")

    class Config:
        json_schema_extra = {
            "example": {
                "Pregnancies": 6,
                "Glucose": 148,
                "BloodPressure": 72,
                "SkinThickness": 35,
                "Insulin": 0,
                "BMI": 33.6,
                "DiabetesPedigreeFunction": 0.627,
                "Age": 50,
            }
        }


class PredictionOutput(BaseModel):
    prediction: int
    probability: float
    risk_label: str


# ----------------------------------------------------------------------
# Routes
# ------------------------------------------------------
# ----------------
@app.get("/")
def root():
    return {"message": "Diabetes Prediction API is running. Visit /docs for usage."}


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: DiabetesInput):
    # Convert validated input into the exact column order the model was trained on
    features = np.array([[
        input_data.Pregnancies,
        input_data.Glucose,
        input_data.BloodPressure,
        input_data.SkinThickness,
        input_data.Insulin,
        input_data.BMI,
        input_data.DiabetesPedigreeFunction,
        input_data.Age,
    ]])

    # Apply the same scaling used during training
    scaled_features = scaler.transform(features)

    # Predict
    prediction = int(model.predict(scaled_features)[0])
    probability = float(model.predict_proba(scaled_features)[0][1])

    risk_label = "High risk" if prediction == 1 else "Low risk"

    return PredictionOutput(
        prediction=prediction,
        probability=round(probability, 4),
        risk_label=risk_label,
    )


