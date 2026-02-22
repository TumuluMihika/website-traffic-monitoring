# src/forecasting.py

"""
Forecasting module using SARIMA (Seasonal ARIMA)
for long-term website traffic prediction.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---------------------------------------------------
# TRAIN SARIMA MODEL
# ---------------------------------------------------

def train_sarima_model(train_df, order=(1,1,1), seasonal_order=(1,1,1,7)):
    """
    Train SARIMA model on a clean univariate time series.
    """

    # Ensure 1D Series
    if isinstance(train_df, pd.DataFrame):
        train_series = train_df.iloc[:, 0]
    else:
        train_series = train_df

    # Convert to numeric
    train_series = pd.to_numeric(train_series, errors="coerce")

    # Strong NaN handling
    train_series = train_series.fillna(method="ffill")
    train_series = train_series.fillna(method="bfill")
    train_series = train_series.dropna()

    # Prevent empty training
    if len(train_series) < 10:
        raise ValueError("Training data too small after cleaning.")

    model = SARIMAX(
        train_series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=True,
        enforce_invertibility=True
    )

    fitted_model = model.fit(disp=False)
    return fitted_model


# ---------------------------------------------------
# MAKE FORECAST
# ---------------------------------------------------

def make_forecast(model, steps: int):
    """
    Generate future forecast safely.
    """

    if steps <= 0:
        raise ValueError("Forecast steps must be positive.")

    forecast = model.forecast(steps=steps)

    forecast = pd.to_numeric(forecast, errors="coerce")
    forecast = forecast.fillna(method="ffill")
    forecast = forecast.fillna(method="bfill")

    return forecast


# ---------------------------------------------------
# EVALUATE FORECAST
# ---------------------------------------------------

def evaluate_forecast(model, test_df: pd.DataFrame):
    """
    Evaluate forecast performance using MAE, RMSE, and SMAPE.
    """

    # Ensure 1D
    if isinstance(test_df, pd.DataFrame):
        test_series = test_df.iloc[:, 0]
    else:
        test_series = test_df

    test_series = pd.to_numeric(test_series, errors="coerce")
    test_series = test_series.fillna(method="ffill")
    test_series = test_series.fillna(method="bfill")
    test_series = test_series.dropna()

    if len(test_series) == 0:
        raise ValueError("Test data empty after cleaning.")

    predictions = model.forecast(steps=len(test_series))

    mae = mean_absolute_error(test_series, predictions)
    rmse = np.sqrt(mean_squared_error(test_series, predictions))

    # SMAPE
    smape = np.mean(
        2 * np.abs(predictions.values - test_series.values) /
        (np.abs(test_series.values) + np.abs(predictions.values) + 1e-10)
    ) * 100

    return {
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "SMAPE": round(smape, 3),
    }