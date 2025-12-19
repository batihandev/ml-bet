from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from features_online import build_feature_row_for_live_match
from predict import build_X, load_model


def predict_live_with_history(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    High-level function:
      - build a feature row for a new match using historic data,
      - run HT/FT and FT 1X2 models on it,
      - return JSON-serializable result.

    payload must include at least:
      division, match_date, home_team, away_team, odd_home, odd_draw, odd_away
    """
    # 1) Build feature row (Series)
    row = build_feature_row_for_live_match(payload)
    df_row = row.to_frame().T  # DataFrame with single row

    # Basic echo info for UI
    match_info = {
        "division": row.get("division"),
        "match_date": row.get("match_date").date().isoformat()
        if isinstance(row.get("match_date"), (pd.Timestamp,))
        else None,
        "home_team": row.get("home_team"),
        "away_team": row.get("away_team"),
        "odd_home": float(row.get("odd_home")) if not pd.isna(row.get("odd_home")) else None,
        "odd_draw": float(row.get("odd_draw")) if not pd.isna(row.get("odd_draw")) else None,
        "odd_away": float(row.get("odd_away")) if not pd.isna(row.get("odd_away")) else None,
    }

    # --------------------
    # HT/FT pattern models
    # --------------------
    pattern_models = [
        ("ht_home_ft_home", "model_ht_home_ft_home", "HT HOME + FT HOME"),
        ("ht_home_ft_draw", "model_ht_home_ft_draw", "HT HOME + FT DRAW"),
        ("ht_home_ft_away", "model_ht_home_ft_away", "HT HOME + FT AWAY"),
        ("ht_draw_ft_home", "model_ht_draw_ft_home", "HT DRAW + FT HOME"),
        ("ht_draw_ft_draw", "model_ht_draw_ft_draw", "HT DRAW + FT DRAW"),
        ("ht_draw_ft_away", "model_ht_draw_ft_away", "HT DRAW + FT AWAY"),
        ("ht_away_ft_home", "model_ht_away_ft_home", "HT AWAY + FT HOME"),
        ("ht_away_ft_draw", "model_ht_away_ft_draw", "HT AWAY + FT DRAW"),
        ("ht_away_ft_away", "model_ht_away_ft_away", "HT AWAY + FT AWAY"),
    ]

    htft_rows: List[Dict[str, Any]] = []

    for key, model_name, label in pattern_models:
        try:
            model, feats = load_model(model_name)
        except FileNotFoundError:
            continue

        X = build_X(df_row, feats)
        prob = float(model.predict_proba(X)[0, 1])
        prob = float(np.clip(prob, 0.001, 0.999))

        htft_rows.append({"key": key, "label": label, "prob": prob})

    # -------------
    # FT 1X2 models
    # -------------
    ft_models = [
        ("ft_home_win", "model_ft_home_win", "Home win (1)"),
        ("ft_draw", "model_ft_draw", "Draw (X)"),
        ("ft_away_win", "model_ft_away_win", "Away win (2)"),
    ]

    ft_rows: List[Dict[str, Any]] = []

    for key, model_name, label in ft_models:
        try:
            model, feats = load_model(model_name)
        except FileNotFoundError:
            continue

        X = build_X(df_row, feats)
        prob = float(model.predict_proba(X)[0, 1])
        prob = float(np.clip(prob, 0.001, 0.999))

        ft_rows.append({"key": key, "label": label, "prob": prob})

    return {
        "match": match_info,
        "htft": htft_rows,
        "ft_1x2": ft_rows,
    }
