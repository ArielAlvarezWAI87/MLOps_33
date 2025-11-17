"""
api.py - FastAPI service for steel energy consumption prediction
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import logging
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.model_prediction import RuleFitPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic Models
class EnergyDataInput(BaseModel):
    """Input schema for energy data"""
    date: str = Field(..., description="Timestamp", example="2018-01-15 14:30:00")
    Lagging_Current_Reactive_Power_kVarh: float = Field(..., ge=0, le=100)
    Leading_Current_Reactive_Power_kVarh: float = Field(..., ge=0, le=30)
    CO2_tCO2: float = Field(..., ge=0, le=0.02, alias="CO2(tCO2)")
    Lagging_Current_Power_Factor: float = Field(..., ge=0, le=1)
    Leading_Current_Power_Factor: float = Field(..., ge=0, le=1)
    NSM: int = Field(..., ge=0, le=90000)
    WeekStatus: str
    Day_of_week: str
    Load_Type: str

    @validator('WeekStatus')
    def validate_week_status(cls, v):
        allowed = ['Weekday', 'Weekend']
        if v not in allowed:
            raise ValueError(f'WeekStatus must be one of {allowed}')
        return v

    @validator('Load_Type')
    def validate_load_type(cls, v):
        allowed = ['Light_Load', 'Medium_Load', 'Maximum_Load']
        if v not in allowed:
            raise ValueError(f'Load_Type must be one of {allowed}')
        return v

    class Config:
        allow_population_by_field_name = True


class PredictionRequest(BaseModel):
    """Request body for predictions"""
    data: List[EnergyDataInput] = Field(..., min_items=1, max_items=1000)


class PredictionResponse(BaseModel):
    """Response schema for predictions"""
    predictions: List[float]
    model_version: str
    prediction_timestamp: str
    n_predictions: int


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    preprocessor_loaded: bool
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Model metadata response"""
    model_name: str
    model_version: str
    model_path: str
    preprocessor_path: str
    model_type: str


# FastAPI Application
app = FastAPI(
    title="Steel Energy Prediction API",
    description="REST API for predicting steel energy consumption",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global variables
MODEL_NAME = "rulefit_steel_energy"
MODEL_VERSION = "1.0.0"
MODEL_PATH = None
PREPROCESSOR_PATH = None
predictor = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global predictor, MODEL_PATH, PREPROCESSOR_PATH

    try:
        ROOT_DIR = Path(__file__).resolve().parents[2]
        MODEL_PATH = ROOT_DIR / "models" / "rulefit.pkl"
        PREPROCESSOR_PATH = ROOT_DIR / "data" / "processed" / "preprocessor.pkl"

        logger.info(f"Loading model from: {MODEL_PATH}")
        predictor = RuleFitPredictor(str(MODEL_PATH), str(PREPROCESSOR_PATH))
        predictor.load_model()
        predictor.load_preprocessor()
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Failed to initialize: {e}")


@app.get("/")
async def root():
    """API information"""
    return {
        "name": "Steel Energy Prediction API",
        "version": MODEL_VERSION,
        "model": MODEL_NAME,
        "endpoints": {
            "health": "/health",
            "model_info": "/model/info",
            "predict": "/predict",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check"""
    return HealthResponse(
        status="healthy" if predictor else "unhealthy",
        model_loaded=predictor is not None and predictor.model is not None,
        preprocessor_loaded=predictor is not None and predictor.preprocessor is not None,
        timestamp=datetime.now().isoformat()
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get model metadata"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfoResponse(
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        model_path=str(MODEL_PATH),
        preprocessor_path=str(PREPROCESSOR_PATH),
        model_type="RuleFitRegressor"
    )


def engineer_features_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering for inference (without Usage_kWh dependency)"""
    df_eng = df.copy()

    # Parse date column
    df_eng["date"] = pd.to_datetime(df_eng["date"])

    # Temporal features
    df_eng["year"] = df_eng["date"].dt.year
    df_eng["month"] = df_eng["date"].dt.month
    df_eng["day"] = df_eng["date"].dt.day
    df_eng["hour"] = df_eng["date"].dt.hour
    df_eng["day_of_week_num"] = df_eng["date"].dt.dayofweek
    df_eng["quarter"] = df_eng["date"].dt.quarter
    df_eng["is_weekend"] = (df_eng["day_of_week_num"] >= 5).astype(int)

    # Cyclical features
    df_eng["hour_sin"] = np.sin(2 * np.pi * df_eng["hour"] / 24)
    df_eng["hour_cos"] = np.cos(2 * np.pi * df_eng["hour"] / 24)
    df_eng["month_sin"] = np.sin(2 * np.pi * df_eng["month"] / 12)
    df_eng["month_cos"] = np.cos(2 * np.pi * df_eng["month"] / 12)
    df_eng["dow_sin"] = np.sin(2 * np.pi * df_eng["day_of_week_num"] / 7)
    df_eng["dow_cos"] = np.cos(2 * np.pi * df_eng["day_of_week_num"] / 7)

    # Power factor features
    df_eng["power_factor_ratio"] = df_eng["Lagging_Current_Power_Factor"] / (
        df_eng["Leading_Current_Power_Factor"] + 1e-6
    )
    df_eng["power_factor_diff"] = (
        df_eng["Lagging_Current_Power_Factor"] - df_eng["Leading_Current_Power_Factor"]
    )
    df_eng["avg_power_factor"] = (
        df_eng["Lagging_Current_Power_Factor"] + df_eng["Leading_Current_Power_Factor"]
    ) / 2

    # Reactive power features
    df_eng["reactive_power_total"] = (
        df_eng["Lagging_Current_Reactive.Power_kVarh"] +
        df_eng["Leading_Current_Reactive_Power_kVarh"]
    )
    df_eng["reactive_power_diff"] = (
        df_eng["Lagging_Current_Reactive.Power_kVarh"] -
        df_eng["Leading_Current_Reactive_Power_kVarh"]
    )
    df_eng["reactive_power_ratio"] = (
        df_eng["Lagging_Current_Reactive.Power_kVarh"] /
        (df_eng["Leading_Current_Reactive_Power_kVarh"] + 1e-6)
    )

    # Energy efficiency indicators (without Usage_kWh)
    # Use median NSM as placeholder for ratios
    median_usage = 75.0  # Approximate median from training data
    df_eng["co2_per_kwh"] = df_eng["CO2(tCO2)"] / median_usage
    df_eng["is_high_consumption"] = 0  # Will be updated by model
    df_eng["nsm_per_kwh"] = df_eng["NSM"] / median_usage

    # Drop date column and Day_of_week (not used in model)
    exclude_cols = ["date", "Day_of_week"]
    df_eng = df_eng.drop(columns=[c for c in exclude_cols if c in df_eng.columns])

    return df_eng


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions"""
    if predictor is None or predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert to DataFrame
        data_dict = [item.dict(by_alias=True) for item in request.data]
        df = pd.DataFrame(data_dict)

        # Rename CO2 column
        if 'CO2(tCO2)' in df.columns:
            pass
        elif 'CO2_tCO2' in df.columns:
            df = df.rename(columns={'CO2_tCO2': 'CO2(tCO2)'})

        logger.info(f"Received prediction request for {len(df)} data points")

        # Rename columns to match training data format
        column_mapping = {
            'Lagging_Current_Reactive_Power_kVarh': 'Lagging_Current_Reactive.Power_kVarh'
        }
        df = df.rename(columns=column_mapping)

        # Step 1: Engineer features
        df_engineered = engineer_features_for_inference(df)

        # Step 2: Transform using preprocessor
        X_transformed = predictor.preprocessor.transform(df_engineered)

        # Step 3: Make predictions
        predictions = predictor.model.predict(X_transformed)

        logger.info(f"Generated {len(predictions)} predictions")

        return PredictionResponse(
            predictions=predictions.tolist(),
            model_version=MODEL_VERSION,
            prediction_timestamp=datetime.now().isoformat(),
            n_predictions=len(predictions)
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(ve)}")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
