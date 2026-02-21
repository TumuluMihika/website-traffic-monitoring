import pandas as pd
import matplotlib.pyplot as plt
from src.config import TIMESTAMP_COL, TARGET_COL

def basic_info(df: pd.DataFrame) -> None:
    print("\n--- Dataset Info ---")
    print(df.info())

    print("\n--- Missing Values ---")
    print(df.isnull().sum())

def convert_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    return df

def traffic_summary(df: pd.DataFrame) -> None:
    print("\n--- Traffic Summary ---")
    print(df[TARGET_COL].describe())

def plot_raw_traffic(df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(df[TIMESTAMP_COL], df[TARGET_COL])
    plt.xlabel("Time")
    plt.ylabel("Traffic Count")
    plt.title("Website Traffic Over Time")
    plt.tight_layout()
    plt.show()
