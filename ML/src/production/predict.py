import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from .schema import CLASS_MAPPING, Outcome, MIN_EDGE, MIN_EV
from features import build_features
from .utils_market import is_valid_odds, calculate_implied_probs

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
    return bundle.get("base_model"), bundle.get("calibrator"), bundle["features"]

def predict_ft_1x2(df_features: pd.DataFrame) -> List[Dict[str, Any]]:
    """Perform production inference on a dataframe of features."""
    base_model, calibrator, features = load_production_model()
    
    # Ensure all required features exist
    X = df_features.copy()
    missing = [c for c in features if c not in X.columns]
    for c in missing:
        X[c] = 0.0
    X = X[features].fillna(0.0)
    
    # Use base model to get raw probabilities
    p_raw = base_model.predict_proba(X)
    
    # Use calibrator if available
    if calibrator:
        if hasattr(calibrator, "transform"):
            probs = calibrator.transform(p_raw)
        elif hasattr(calibrator, "predict_proba_from_raw"):
            probs = calibrator.predict_proba_from_raw(p_raw)
        else:
            # Fallback if it's a standard sklearn calibrator
            probs = calibrator.predict_proba(X)
    else:
        probs = p_raw
    
    results = []
    for i in range(len(df_features)):
        row = df_features.iloc[i]
        p_home, p_draw, p_away = probs[i]
        
        odd_home = row.get("odd_home")
        odd_draw = row.get("odd_draw")
        odd_away = row.get("odd_away")
        
        # Centralized check: returns tuple or None
        implied = calculate_implied_probs(odd_home, odd_draw, odd_away)
        
        metrics = []
        if implied is not None:
            pimp_home, pimp_draw, pimp_away = implied
            
            # Recast as float for safety
            oh = float(odd_home)
            od = float(odd_draw)
            oa = float(odd_away)
            
            metrics.append({
                "outcome": "home",
                "prob": float(p_home),
                "pimp": float(pimp_home),
                "edge": float(p_home - pimp_home),
                "ev": float(p_home * oh - 1),
                "odds": oh
            })
            metrics.append({
                "outcome": "draw",
                "prob": float(p_draw),
                "pimp": float(pimp_draw),
                "edge": float(p_draw - pimp_draw),
                "ev": float(p_draw * od - 1),
                "odds": od
            })
            metrics.append({
                "outcome": "away",
                "prob": float(p_away),
                "pimp": float(pimp_away),
                "edge": float(p_away - pimp_away),
                "ev": float(p_away * oa - 1),
                "odds": oa
            })
        
        results.append({
            "match_id": int(row["match_id"]),
            "probabilities": {
                "home": float(p_home),
                "draw": float(p_draw),
                "away": float(p_away)
            },
            "implied_probabilities": {
                "home": float(implied[0]) if implied else None,
                "draw": float(implied[1]) if implied else None,
                "away": float(implied[2]) if implied else None
            },
            "metrics": metrics
        })
        
    return results

def predict_live_with_history(
    payload: Dict[str, Any], 
    min_edge: float = MIN_EDGE, 
    min_ev: float = MIN_EV
) -> Dict[str, Any]:
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
        "odd_home": float(row.get("odd_home")) if is_valid_odds(row.get("odd_home")) else None,
        "odd_draw": float(row.get("odd_draw")) if is_valid_odds(row.get("odd_draw")) else None,
        "odd_away": float(row.get("odd_away")) if is_valid_odds(row.get("odd_away")) else None,
    }
    
    res = results[0]
    
    # Apply default decision logic for live/UI usage
    recommendation = "no bet"
    best_ev = -999.0
    metrics = res["metrics"]
    
    if metrics:
        best_choice = max(metrics, key=lambda x: x["ev"])
        # Gate with both thresholds
        if best_choice["ev"] >= min_ev and best_choice["edge"] >= min_edge:
            recommendation = best_choice["outcome"]
            best_ev = best_choice["ev"]
            
    return {
        "match": match_info,
        "probabilities": res["probabilities"],
        "implied_probabilities": res["implied_probabilities"],
        "recommendation": recommendation,
        "best_ev": float(best_ev) if recommendation != "no bet" else None,
        "metrics": metrics
    }
