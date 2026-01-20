import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List
from features import build_features
from .engine import build_X, load_model

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

def _load_processed_matches() -> pd.DataFrame:
    path = PROCESSED_DIR / "matches.csv"
    if not path.exists():
        raise FileNotFoundError(f"Processed matches not found: {path}")
    return pd.read_csv(path, parse_dates=["match_date"], low_memory=False)

def build_feature_row_for_live_match(payload: Dict[str, Any]) -> pd.Series:
    df_hist = _load_processed_matches()
    max_id = pd.to_numeric(df_hist.get("match_id", pd.Series([], dtype="float64")), errors="coerce").max()
    next_id = int(max_id + 1) if np.isfinite(max_id) else int(len(df_hist))

    live_row = {
        "match_id": next_id, "is_synthetic": 1,
        "division": payload["division"], "match_date": pd.to_datetime(payload["match_date"]),
        "home_team": payload["home_team"], "away_team": payload["away_team"],
        "odd_home": payload.get("odd_home"), "odd_draw": payload.get("odd_draw"), "odd_away": payload.get("odd_away"),
    }
    for col in ["ht_home_goals", "ht_away_goals", "ft_home_goals", "ft_away_goals"]:
        live_row[col] = np.nan

    for k in live_row.keys():
        if k not in df_hist.columns: df_hist[k] = np.nan

    df_aug = pd.concat([df_hist, pd.DataFrame([live_row])], ignore_index=True)
    df_feat = build_features(df_aug)
    row = df_feat[df_feat["match_id"] == next_id]
    if row.empty: raise ValueError("build_features() failed for synthetic row.")
    return row.iloc[0]

def predict_live_with_history(payload: Dict[str, Any]) -> Dict[str, Any]:
    row = build_feature_row_for_live_match(payload)
    df_row = row.to_frame().T

    match_info = {
        "division": row.get("division"),
        "match_date": row.get("match_date").date().isoformat() if isinstance(row.get("match_date"), pd.Timestamp) else None,
        "home_team": row.get("home_team"), "away_team": row.get("away_team"),
        "odd_home": float(row.get("odd_home")) if not pd.isna(row.get("odd_home")) else None,
        "odd_draw": float(row.get("odd_draw")) if not pd.isna(row.get("odd_draw")) else None,
        "odd_away": float(row.get("odd_away")) if not pd.isna(row.get("odd_away")) else None,
    }

    models_config = {
        "htft": [
            ("ht_home_ft_home", "model_ht_home_ft_home", "HT HOME + FT HOME"),
            ("ht_home_ft_draw", "model_ht_home_ft_draw", "HT HOME + FT DRAW"),
            ("ht_home_ft_away", "model_ht_home_ft_away", "HT HOME + FT AWAY"),
            ("ht_draw_ft_home", "model_ht_draw_ft_home", "HT DRAW + FT HOME"),
            ("ht_draw_ft_draw", "model_ht_draw_ft_draw", "HT DRAW + FT DRAW"),
            ("ht_draw_ft_away", "model_ht_draw_ft_away", "HT DRAW + FT AWAY"),
            ("ht_away_ft_home", "model_ht_away_ft_home", "HT AWAY + FT HOME"),
            ("ht_away_ft_draw", "model_ht_away_ft_draw", "HT AWAY + FT DRAW"),
            ("ht_away_ft_away", "model_ht_away_ft_away", "HT AWAY + FT AWAY"),
        ],
        "ft_1x2": [
            ("ft_home_win", "model_ft_home_win", "Home win (1)"),
            ("ft_draw", "model_ft_draw", "Draw (X)"),
            ("ft_away_win", "model_ft_away_win", "Away win (2)"),
        ]
    }

    results = {"match": match_info, "htft": [], "ft_1x2": []}
    for category, models in models_config.items():
        for key, model_name, label in models:
            try:
                model, feats = load_model(model_name)
                X = build_X(df_row, feats)
                prob = float(np.clip(model.predict_proba(X)[0, 1], 0.001, 0.999))
                results[category].append({"key": key, "label": label, "prob": prob})
            except FileNotFoundError:
                continue
    return results
