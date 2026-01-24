import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from dataset.cleaner import make_time_split, select_feature_columns, clean_training_data

ROOT_DIR = Path(__file__).resolve().parents[3] # experimental/training/data.py -> src -> ML
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

def load_features() -> pd.DataFrame:
    path = PROCESSED_DIR / "features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")
    df = pd.read_csv(path, parse_dates=["match_date"], low_memory=False, dtype={"match_time": "string"})
    return df
