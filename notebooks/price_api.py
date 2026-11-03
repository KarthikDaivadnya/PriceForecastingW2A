
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import joblib
import json
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "price_forecasting_rf_model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

model = joblib.load(MODEL_PATH)

with open(METADATA_PATH, "r") as file:
    metadata = json.load(file)

app = FastAPI(
    title="Monthly Price Forecasting API",
    description="AI-based monthly price prediction service",
    version="1.0.0"
)


class PredictionRequest(BaseModel):
    historical_prices: list[float]
    last_date: str


def create_features(history, date):
    features = {
        "month": date.month,
        "year": date.year,
        "month_sin": np.sin(2 * np.pi * date.month / 12),
        "month_cos": np.cos(2 * np.pi * date.month / 12)
    }

    for lag in [1, 2, 3, 4, 6, 9, 12]:
        features[f"lag_{lag}"] = history.iloc[-lag]

    for window in [3, 6, 12]:
        features[f"rolling_mean_{window}"] = (
            history.iloc[-window:].mean()
        )

    features["rolling_median_3"] = history.iloc[-3:].median()
    features["rolling_median_6"] = history.iloc[-6:].median()
    features["rolling_std_3"] = history.iloc[-3:].std()
    features["rolling_std_6"] = history.iloc[-6:].std()
    features["rolling_min_12"] = history.iloc[-12:].min()
    features["rolling_max_12"] = history.iloc[-12:].max()

    return pd.DataFrame([features])[metadata["features"]]


def generate_forecast(history, last_date, periods=12):
    if len(history) < 12:
        raise ValueError("At least 12 historical prices are required.")

    history = pd.Series(history, dtype=float)
    history.index = pd.date_range(
        end=pd.Timestamp(last_date),
        periods=len(history),
        freq="MS"
    )

    forecasts = []

    for date in pd.date_range(
        start=history.index[-1] + pd.offsets.MonthBegin(1),
        periods=periods,
        freq="MS"
    ):
        X = create_features(history, date)
        prediction = float(model.predict(X)[0])

        forecasts.append({
            "date": date.strftime("%Y-%m-%d"),
            "predicted_price": round(prediction, 2)
        })

        history.loc[date] = prediction

    return forecasts


@app.get("/")
def home():
    return {
        "message": "Monthly Price Forecasting API is running",
        "status": "healthy"
    }


@app.get("/model-info")
def model_info():
    return metadata


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        if not all(np.isfinite(request.historical_prices)):
            raise ValueError("Historical prices must be finite numbers.")

        forecast = generate_forecast(
            request.historical_prices,
            request.last_date
        )

        return {
            "model": metadata["model_name"],
            "forecast_periods": len(forecast),
            "forecast": forecast
        }

    except (ValueError, IndexError) as error:
        raise HTTPException(status_code=400, detail=str(error))
