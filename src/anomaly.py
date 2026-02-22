# src/anomaly.py

import pandas as pd
import numpy as np

def detect_anomalies(model, test_df, threshold_multiplier=2.0):

    # Always force 1D series
    if isinstance(test_df, pd.DataFrame):
        test_series = test_df.iloc[:, 0]
    else:
        test_series = test_df

    test_series = pd.to_numeric(test_series, errors="coerce")
    test_series = test_series.fillna(method="ffill")

    # Forecast EXACT same length as test_df
    predictions = model.forecast(steps=len(test_series))

    residuals = test_series.values - predictions.values

    std_dev = np.std(residuals)
    threshold = threshold_multiplier * std_dev

    results = pd.DataFrame({
        "Actual": test_series.values,
        "Predicted": predictions.values,
        "Residual": residuals
    }, index=test_series.index)

    results["Anomaly"] = abs(results["Residual"]) > threshold

    anomaly_count = int(results["Anomaly"].sum())

    return results, anomaly_count