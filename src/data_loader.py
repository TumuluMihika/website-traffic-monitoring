import pandas as pd
from src.config import DATA_PATH

def load_traffic_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df
