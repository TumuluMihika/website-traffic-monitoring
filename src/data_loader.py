import pandas as pd
from src.config import DATASET_TYPE, DATA_PATHS

def load_traffic_data():
    path = DATA_PATHS[DATASET_TYPE]
    df = pd.read_csv(path)

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

    # 🔥 FORCE NUMERIC CONVERSION (CRITICAL FIX)
    df["Page.Loads"] = (
        df["Page.Loads"]
        .astype(str)
        .str.replace(",", "", regex=False)
    )

    df["Page.Loads"] = pd.to_numeric(df["Page.Loads"], errors="coerce")

    return df