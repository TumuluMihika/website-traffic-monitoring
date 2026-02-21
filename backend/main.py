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

    # Preprocessing
    df = convert_timestamp(df)
    df = set_time_index(df)
    df = handle_missing_values(df)

    daily_df = resample_daily(df)
    train_df, test_df = train_test_split(daily_df, test_days=test_days)

    model = train_sarima_model(train_df)
    forecast = make_forecast(model, steps=len(test_df))

    # Limit visualization
    train_df = train_df.last("3Y")

    # Detect anomalies
    anomaly_results, anomaly_count = detect_anomalies(model, test_df)
    anomalies = anomaly_results[anomaly_results["Anomaly"]]

    # -----------------------------
    # Severity Calculation
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

    return {
        # Graph Data
        "historical_dates": train_df.index.strftime("%Y-%m-%d").tolist(),
        "historical_values": train_df.squeeze().tolist(),
        "forecast_dates": forecast.index.strftime("%Y-%m-%d").tolist(),
        "forecast_values": forecast.tolist(),

        # Anomaly Data
        "anomaly_count": int(anomaly_count),
        "anomaly_dates": anomalies.index.strftime("%Y-%m-%d").tolist(),
        "anomaly_actual": anomalies["Actual"].tolist(),
        "anomaly_predicted": anomalies["Predicted"].tolist(),
        "anomaly_residual": anomalies["Residual"].round(2).tolist(),
        "severity": severity_levels
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