import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

def load_features() -> pd.DataFrame:
    path = PROCESSED_DIR / "features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")
    df = pd.read_csv(path, parse_dates=["match_date"], low_memory=False, dtype={"match_time": "string"})
    return df

def make_time_split(df: pd.DataFrame, fixed_cutoff: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    if fixed_cutoff:
        cutoff = pd.to_datetime(fixed_cutoff)
    else:
        cutoff = df["match_date"].quantile(0.8)
    train_mask = df["match_date"] < cutoff
    val_mask = ~train_mask
    return train_mask.values, val_mask.values

def select_feature_columns(df: pd.DataFrame) -> List[str]:
    base_features = [
        "odd_home", "odd_draw", "odd_away", "max_odd_home", "max_odd_draw", "max_odd_away",
        "odd_over25", "odd_under25", "max_odd_over25", "max_odd_under25",
        "handicap_size", "handicap_home", "handicap_away",
    ]
    league_features = [
        "league_avg_goals", "league_home_win_rate", "league_draw_rate", "league_away_win_rate",
        "home_attack_strength", "away_attack_strength",
    ]
    congestion_features = [
        "home_days_since_last", "away_days_since_last", "home_matches_last_7d", "home_matches_last_14d",
        "home_matches_last_21d", "away_matches_last_7d", "away_matches_last_14d", "away_matches_last_21d",
    ]
    rolling_features = [c for c in df.columns if c.startswith("home_form") or c.startswith("away_form")]
    h2h_features = [
        "h2h_htd_ft_home_win_rate", "h2h_htd_ft_away_win_rate", "h2h_matches_count",
        "rule_score_home_htdftw", "rule_score_away_htdftw",
    ]
    
    feature_cols = []
    for c in base_features + league_features + congestion_features + h2h_features:
        if c in df.columns: feature_cols.append(c)
    feature_cols.extend(sorted(rolling_features))
    return feature_cols

def clean_training_data(df: pd.DataFrame, feature_cols: List[str], target_col: str, fixed_cutoff: Optional[str] = None):
    # Filter rows where target is missing
    df = df[~df[target_col].isna()].copy()
    
    # Require at least some history for rolling features
    rolling_cols = [c for c in feature_cols if ("form" in c) or c.startswith("h2h_")]
    if rolling_cols:
        has_some_history = df[rolling_cols].notna().any(axis=1)
        df = df[has_some_history].copy()
    
    X = df[feature_cols].copy().fillna(0.0)
    y = df[target_col].astype(int).values
    train_mask, val_mask = make_time_split(df, fixed_cutoff)
    
    return X[train_mask], X[val_mask], y[train_mask], y[val_mask], feature_cols
