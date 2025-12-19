from __future__ import annotations

from typing import Any, Dict
from pathlib import Path

import numpy as np
import pandas as pd

from features import build_features  # <- now exists

ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


def _load_processed_matches() -> pd.DataFrame:
    path = PROCESSED_DIR / "matches.csv"
    if not path.exists():
        raise FileNotFoundError(f"Processed matches not found: {path}")
    return pd.read_csv(path, parse_dates=["match_date"], low_memory=False)


def build_feature_row_for_live_match(payload: Dict[str, Any]) -> pd.Series:
    df_hist = _load_processed_matches()

    # Create a unique synthetic id
    max_id = pd.to_numeric(df_hist.get("match_id", pd.Series([], dtype="float64")), errors="coerce").max()
    next_id = int(max_id + 1) if np.isfinite(max_id) else int(len(df_hist))

    live_row = {
        "match_id": next_id,
        "is_synthetic": 1,

        "division": payload["division"],
        "match_date": pd.to_datetime(payload["match_date"]),
        "home_team": payload["home_team"],
        "away_team": payload["away_team"],

        # odds you have now
        "odd_home": payload.get("odd_home"),
        "odd_draw": payload.get("odd_draw"),
        "odd_away": payload.get("odd_away"),
    }

    # Ensure all expected columns exist in the historical df
    for k in live_row.keys():
        if k not in df_hist.columns:
            df_hist[k] = np.nan

    # Add required goal columns if your pipeline expects them
    for col in ["ht_home_goals", "ht_away_goals", "ft_home_goals", "ft_away_goals"]:
        if col not in df_hist.columns:
            df_hist[col] = np.nan

    # Future match has unknown goals
    live_row["ht_home_goals"] = np.nan
    live_row["ht_away_goals"] = np.nan
    live_row["ft_home_goals"] = np.nan
    live_row["ft_away_goals"] = np.nan

    df_aug = pd.concat([df_hist, pd.DataFrame([live_row])], ignore_index=True)

    df_feat = build_features(df_aug)

    # Extract by match_id (most reliable)
    row = df_feat[df_feat["match_id"] == next_id]
    if row.empty:
        raise ValueError("build_features() did not produce the synthetic match row.")
    return row.iloc[0]
