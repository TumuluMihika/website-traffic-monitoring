# src/config.py
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "web_traffic.csv")

TIMESTAMP_COL = "Timestamp"
TARGET_COL = "TrafficCount"
