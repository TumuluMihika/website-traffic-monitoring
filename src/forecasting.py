# src/forecasting.py

"""
Forecasting module using SARIMA (Seasonal ARIMA)
for long-term website traffic prediction.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_sarima_model(
    train_df: pd.DataFrame,
    order=(2,1,2),
    seasonal_order=(1,1,1,7)
):
    """
    Train SARIMA model.

    Parameters:
        train_df: Daily traffic dataframe with datetime index
        order: (p,d,q)
        seasonal_order: (P,D,Q,s)

    Returns:
        Fitted SARIMA model
    """

    model = SARIMAX(
        train_df,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fitted_model = model.fit(disp=False)
    return fitted_model


def make_forecast(model, steps: int):
    """
    Generate future forecast.
    """
    forecast = model.forecast(steps=steps)
    return forecast


def evaluate_forecast(model, test_df: pd.DataFrame):
    predictions = model.forecast(steps=len(test_df))

    mae = mean_absolute_error(test_df, predictions)
    rmse = np.sqrt(mean_squared_error(test_df, predictions))

    # SMAPE
    smape = np.mean(
        2 * np.abs(predictions.values - test_df.values) /
        (np.abs(test_df.values) + np.abs(predictions.values) + 1e-10)
    ) * 100
    

    return {
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "SMAPE": round(smape, 3),
    }
