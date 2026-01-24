import joblib
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

def load_features() -> pd.DataFrame:
    path = PROCESSED_DIR / "features.csv"
    if not path.exists(): raise FileNotFoundError(f"Feature file not found: {path}")
    return pd.read_csv(path, parse_dates=["match_date"], low_memory=False, dtype={"match_time": "string"})

def load_model(model_name: str):
    path = MODELS_DIR / f"{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    bundle = joblib.load(path)
    return bundle["model"], bundle["features"]

def build_X(df_src: pd.DataFrame, feature_list: list[str]) -> pd.DataFrame:
    X = df_src.copy()
    missing = [c for c in feature_list if c not in X.columns]
    for c in missing: X[c] = 0.0
    return X[feature_list].copy().fillna(0.0)
