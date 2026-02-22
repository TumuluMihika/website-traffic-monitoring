from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import io
import logging

from src.data_loader import load_traffic_data
from src.data_overview import convert_timestamp
from src.preprocessing import (
    set_time_index,
    handle_missing_values,
    resample_daily,
    train_test_split
)
from src.forecasting import train_sarima_model, make_forecast, evaluate_forecast
from src.anomaly import detect_anomalies


# -----------------------------
# APP SETUP
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)


# -----------------------------
# CORE PIPELINE FUNCTION
# -----------------------------
def run_pipeline(df: pd.DataFrame, test_days: int = 60):

    # -----------------------------
    # PREPROCESSING
    # -----------------------------
    df = convert_timestamp(df)
    df = set_time_index(df)
    df = handle_missing_values(df)

    daily_df = resample_daily(df)
    train_df, test_df = train_test_split(daily_df, test_days=test_days)

    model = train_sarima_model(train_df)
    forecast = make_forecast(model, steps=len(test_df))

    # Limit visualization to last 3 years
    train_df = train_df.last("3Y")

    # -----------------------------
    # ANOMALY DETECTION
    # -----------------------------
    anomaly_results, anomaly_count = detect_anomalies(model, test_df)
    anomalies = anomaly_results[anomaly_results["Anomaly"]]

    # -----------------------------
    # SEVERITY CALCULATION
    # -----------------------------
    severity_levels = []
    abs_residuals = anomaly_results["Residual"].abs()

    if len(abs_residuals) > 0:
        p90 = abs_residuals.quantile(0.90)
        p97 = abs_residuals.quantile(0.97)
    else:
        p90, p97 = 0, 0

    for residual in anomalies["Residual"]:
        abs_res = abs(residual)

        if abs_res < p90:
            severity_levels.append("Low")
        elif abs_res < p97:
            severity_levels.append("Medium")
        else:
            severity_levels.append("High")

    # -----------------------------
    # STRUCTURED ANOMALY OBJECTS
    # -----------------------------
    anomaly_objects = [
        {
            "date": date.strftime("%Y-%m-%d"),
            "actual": float(actual),
            "predicted": float(predicted),
            "residual": float(round(residual, 2)),
            "severity": severity
        }
        for date, actual, predicted, residual, severity in zip(
            anomalies.index,
            anomalies["Actual"],
            anomalies["Predicted"],
            anomalies["Residual"],
            severity_levels
        )
    ]

    return {
        # Graph Data
        "historical_dates": train_df.index.strftime("%Y-%m-%d").tolist(),
        "historical_values": train_df.squeeze().tolist(),
        "forecast_dates": forecast.index.strftime("%Y-%m-%d").tolist(),
        "forecast_values": forecast.tolist(),

        # Clean Anomaly Objects (NEW FORMAT)
        "anomaly_count": int(anomaly_count),
        "anomalies": anomaly_objects
    }

# -----------------------------
# DEFAULT FORECAST
# -----------------------------
@app.get("/forecast")
def get_forecast(days: int = 30):
    try:
        df = load_traffic_data()
        return run_pipeline(df, test_days=days)
    except Exception as e:
        logging.error(f"Forecast error: {e}")
        return JSONResponse(status_code=500, content={"error": "Forecast failed"})


# -----------------------------
# DEFAULT ANOMALIES
# -----------------------------
@app.post("/anomalies")
def get_anomalies():
    try:
        df = load_traffic_data()
        return run_pipeline(df, test_days=60)
    except Exception as e:
        logging.error(f"Anomaly error: {e}")
        return JSONResponse(status_code=500, content={"error": "Anomaly detection failed"})


# -----------------------------
# METRICS
# -----------------------------
@app.get("/metrics")
def get_metrics():
    try:
        df = load_traffic_data()
        df = convert_timestamp(df)
        df = set_time_index(df)
        df = handle_missing_values(df)

        daily_df = resample_daily(df)
        train_df, test_df = train_test_split(daily_df, test_days=60)

        model = train_sarima_model(train_df)
        metrics = evaluate_forecast(model, test_df)

        return metrics

    except Exception as e:
        logging.error(f"Metrics error: {e}")
        return JSONResponse(status_code=500, content={"error": "Metrics calculation failed"})


# -----------------------------
# USER UPLOAD + ANOMALY
# -----------------------------
@app.post("/upload-anomaly")
async def upload_and_detect(file: UploadFile = File(...)):

    if not file.filename.endswith(".csv"):
        return JSONResponse(
            status_code=400,
            content={"error": "Only CSV files are allowed"}
        )

    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        # ---- STANDARDIZE TIMESTAMP ----
        if "Date" in df.columns:
            df.rename(columns={"Date": "Date"}, inplace=True)
        elif "ds" in df.columns:
            df.rename(columns={"ds": "Date"}, inplace=True)
        elif "Timestamp" in df.columns:
            df.rename(columns={"Timestamp": "Date"}, inplace=True)
        else:
            raise ValueError("No valid timestamp column found")

        # ---- STANDARDIZE TARGET ----
        if "Page.Loads" in df.columns:
            df.rename(columns={"Page.Loads": "Page.Loads"}, inplace=True)
        elif "y" in df.columns:
            df.rename(columns={"y": "Page.Loads"}, inplace=True)
        elif "TrafficCount" in df.columns:
            df.rename(columns={"TrafficCount": "Page.Loads"}, inplace=True)
        else:
            raise ValueError("No valid traffic column found")

        # ---- FORCE NUMERIC ----
        # ---- FORCE NUMERIC CLEANING ----
        df["Page.Loads"] = (
            df["Page.Loads"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
        )

        df["Page.Loads"] = pd.to_numeric(df["Page.Loads"], errors="coerce")

        # 🔥 Drop invalid rows
        df = df.dropna(subset=["Page.Loads", "Date"])

        # 🔥 Ensure dataset not empty
        if df.empty:
            raise ValueError("No valid numeric traffic data found after cleaning")

        if df.empty:
            return JSONResponse(
                status_code=400,
                content={"error": "Uploaded file is empty"}
            )

        logging.info(f"Uploaded file processed: {file.filename}")

        return run_pipeline(df, test_days=60)

    except Exception as e:
        logging.error(f"Upload processing error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to process uploaded file"}
        )