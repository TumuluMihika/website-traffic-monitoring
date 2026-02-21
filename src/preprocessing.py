# src/preprocessing.py

import pandas as pd
from src.config import TIMESTAMP_COL, TARGET_COL

def set_time_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Set timestamp column as DataFrame index.
    """
    df = df.set_index(TIMESTAMP_COL)
    df = df.sort_index()
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values using time interpolation.
    """
    df[TARGET_COL] = df[TARGET_COL].interpolate(method="time")
    return df


def resample_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample traffic data to daily frequency (long-term forecasting).
    """
    daily_df = df.resample("D").sum()
    return daily_df


def prepare_prophet_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert dataframe to Prophet-compatible format.
    """
    prophet_df = df.reset_index()
    prophet_df = prophet_df.rename(
        columns={
            TIMESTAMP_COL: "ds",
            TARGET_COL: "y"
        }
    )
    return prophet_df


def train_test_split(
    df: pd.DataFrame,
    test_days: int = 30
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series into train and test sets.
    """
    train = df.iloc[:-test_days]
    test = df.iloc[-test_days:]
    return train, test
