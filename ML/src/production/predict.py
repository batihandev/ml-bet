import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from .schema import CLASS_MAPPING, Outcome, MIN_EDGE, MIN_EV
from features import build_features

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

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

def load_production_model():
    model_path = MODELS_DIR / "model_ft_1x2.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Production model not found: {model_path}")
    
    bundle = joblib.load(model_path)
    return bundle["model"], bundle["features"]

def calculate_implied_probs(odd_home, odd_draw, odd_away):
    """Calculate normalized implied probabilities from bookmaker odds."""
    if not all([odd_home, odd_draw, odd_away]):
        return None, None, None
    
    raw_probs = [1.0/odd_home, 1.0/odd_draw, 1.0/odd_away]
    margin = sum(raw_probs)
    return [p / margin for p in raw_probs]

def predict_ft_1x2(df_features: pd.DataFrame) -> List[Dict[str, Any]]:
    """Perform production inference on a dataframe of features."""
    model, features = load_production_model()
    
    # Ensure all required features exist
    X = df_features.copy()
    missing = [c for c in features if c not in X.columns]
    for c in missing:
        X[c] = 0.0
    X = X[features].fillna(0.0)
    
    probs = model.predict_proba(X) # Shape (N, 3)
    
    results = []
    for i in range(len(df_features)):
        row = df_features.iloc[i]
        p_home, p_draw, p_away = probs[i]
        
        odd_home = float(row.get("odd_home", 0)) if not pd.isna(row.get("odd_home")) else 0
        odd_draw = float(row.get("odd_draw", 0)) if not pd.isna(row.get("odd_draw")) else 0
        odd_away = float(row.get("odd_away", 0)) if not pd.isna(row.get("odd_away")) else 0
        
        pimp_home, pimp_draw, pimp_away = calculate_implied_probs(odd_home, odd_draw, odd_away)
        
        metrics = []
        if pimp_home is not None:
            metrics.append({
                "outcome": "home",
                "prob": float(p_home),
                "pimp": float(pimp_home),
                "edge": float(p_home - pimp_home),
                "ev": float(p_home * odd_home - 1),
                "odds": float(odd_home)
            })
            metrics.append({
                "outcome": "draw",
                "prob": float(p_draw),
                "pimp": float(pimp_draw),
                "edge": float(p_draw - pimp_draw),
                "ev": float(p_draw * odd_draw - 1),
                "odds": float(odd_draw)
            })
            metrics.append({
                "outcome": "away",
                "prob": float(p_away),
                "pimp": float(pimp_away),
                "edge": float(p_away - pimp_away),
                "ev": float(p_away * odd_away - 1),
                "odds": float(odd_away)
            })
        
        recommendation = "no bet"
        best_ev = -999.0
        if metrics:
            best_choice = max(metrics, key=lambda x: x["ev"])
            if best_choice["ev"] >= MIN_EV and best_choice["edge"] >= MIN_EDGE:
                recommendation = best_choice["outcome"]
                best_ev = best_choice["ev"]
        
        results.append({
            "match_id": int(row["match_id"]),
            "probabilities": {
                "home": float(p_home),
                "draw": float(p_draw),
                "away": float(p_away)
            },
            "implied_probabilities": {
                "home": float(pimp_home) if pimp_home else None,
                "draw": float(pimp_draw) if pimp_draw else None,
                "away": float(pimp_away) if pimp_away else None
            },
            "recommendation": recommendation,
            "best_ev": float(best_ev) if recommendation != "no bet" else None,
            "metrics": metrics
        })
        
    return results

def predict_live_with_history(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Single-model production live prediction."""
    row = build_feature_row_for_live_match(payload)
    df_row = row.to_frame().T
    
    results = predict_ft_1x2(df_row)
    if not results:
        raise ValueError("No prediction generated")
        
    match_info = {
        "division": row.get("division"),
        "match_date": row.get("match_date").date().isoformat() if isinstance(row.get("match_date"), pd.Timestamp) else None,
        "home_team": row.get("home_team"), "away_team": row.get("away_team"),
        "odd_home": float(row.get("odd_home")) if not pd.isna(row.get("odd_home")) else None,
        "odd_draw": float(row.get("odd_draw")) if not pd.isna(row.get("odd_draw")) else None,
        "odd_away": float(row.get("odd_away")) if not pd.isna(row.get("odd_away")) else None,
    }
    
    res = results[0]
    return {
        "match": match_info,
        "probabilities": res["probabilities"],
        "implied_probabilities": res["implied_probabilities"],
        "recommendation": res["recommendation"],
        "best_ev": res["best_ev"],
        "metrics": res["metrics"]
    }
