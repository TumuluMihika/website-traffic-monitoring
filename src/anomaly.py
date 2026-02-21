# src/anomaly.py

import pandas as pd
import numpy as np


def detect_anomalies(model, test_df: pd.DataFrame, threshold_multiplier: float = 1.2):
    """
    Detect anomalies using residual thresholding.
    """

    # Generate predictions
    predictions = model.forecast(steps=len(test_df))

    # Ensure predictions have same index
    predictions.index = test_df.index

    # If test_df is DataFrame, convert to Series
    if isinstance(test_df, pd.DataFrame):
        actual = test_df.squeeze()
    else:
        actual = test_df

    # Compute residuals
    residuals = actual - predictions

    # Compute thresholds
    abs_residuals = np.abs(residuals)
    threshold = np.percentile(abs_residuals, 95)

    upper_threshold = threshold
    lower_threshold = -threshold

    # Create clean result dataframe
    results = pd.DataFrame({
        "Actual": actual,
        "Predicted": predictions,
        "Residual": residuals
    })

    # Flag anomalies
    results["Anomaly"] = (
        (results["Residual"] > upper_threshold) |
        (results["Residual"] < lower_threshold)
    )

    anomaly_count = results["Anomaly"].sum()

    return results, anomaly_count

