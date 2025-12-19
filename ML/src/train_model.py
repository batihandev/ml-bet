import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

# ----------------------------------------------------------------------
# Date controls
#   TRAIN_START_DATE / TRAIN_END_DATE can also be set from environment.
#   If env var is missing or empty, falls back to None.
# ----------------------------------------------------------------------
TRAIN_START_DATE = os.getenv("TRAIN_START_DATE") or None      # e.g. "2020-01-01"
TRAIN_END_DATE = os.getenv("TRAIN_END_DATE") or None          # e.g. "2025-06-01"
FIXED_CUTOFF_DATE = os.getenv("FIXED_CUTOFF_DATE") or None    # e.g. "2022-01-01"


def load_features() -> pd.DataFrame:
    path = PROCESSED_DIR / "features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")

    df = pd.read_csv(
        path,
        parse_dates=["match_date"],
        low_memory=False,
        dtype={"match_time": "string"},
    )
    return df


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Select feature columns to use in training.

    We combine:
      - Odds / handicaps (“base features”)
      - League context (avg goals, home/draw/away win rates, attack strength)
      - Schedule/rest features (days since last, matches in last X days)
      - All home_* / away_* team-form features produced by features.py
        (multi-horizon: form3, form5, form10, formAll)
      - H2H + rule-score features
    """

    # Explicitly allowed base columns (non-rolling)
    base_features = [
        "odd_home",
        "odd_draw",
        "odd_away",
        "max_odd_home",
        "max_odd_draw",
        "max_odd_away",
        "odd_over25",
        "odd_under25",
        "max_odd_over25",
        "max_odd_under25",
        "handicap_size",
        "handicap_home",
        "handicap_away",
    ]

    # League-level context and derived attack strength
    league_features = [
        "league_avg_goals",
        "league_home_win_rate",
        "league_draw_rate",
        "league_away_win_rate",
        "home_attack_strength",
        "away_attack_strength",
    ]

    # Schedule / congestion / rest features
    congestion_features = [
        "home_days_since_last",
        "away_days_since_last",
        "home_matches_last_7d",
        "home_matches_last_14d",
        "home_matches_last_21d",
        "away_matches_last_7d",
        "away_matches_last_14d",
        "away_matches_last_21d",
    ]

    # All rolling form features created in features.py:
    # e.g. home_form3_ft_win_rate, away_form10_goals_for_avg, home_formAll_htd_ftw_rate, ...
    rolling_features = [
        c
        for c in df.columns
        if c.startswith("home_form") or c.startswith("away_form")
    ]

    # H2H and rule-based features
    h2h_features = [
        "h2h_htd_ft_home_win_rate",
        "h2h_htd_ft_away_win_rate",
        "h2h_matches_count",
        "rule_score_home_htdftw",
        "rule_score_away_htdftw",
    ]

    feature_cols: list[str] = []

    # add base + league + congestion + h2h/rule features if present
    for c in base_features + league_features + congestion_features + h2h_features:
        if c in df.columns:
            feature_cols.append(c)

    # add all rolling form features
    feature_cols.extend(sorted(rolling_features))

    return feature_cols


def make_time_split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Split into train/validation by time.

    If FIXED_CUTOFF_DATE is set:
        train < cutoff, val >= cutoff
    Else:
        cutoff = 80% quantile of match_date
    """
    if FIXED_CUTOFF_DATE:
        cutoff = pd.to_datetime(FIXED_CUTOFF_DATE)
    else:
        cutoff = df["match_date"].quantile(0.8)

    train_mask = df["match_date"] < cutoff
    val_mask = ~train_mask
    print(f"Using cutoff date: {cutoff.date()} -> train={train_mask.sum()}, val={val_mask.sum()}")
    return train_mask.values, val_mask.values


def clean_training_data(df: pd.DataFrame, feature_cols: list[str], target_col: str):
    """
    - Drop rows with missing target.
    - Require that at least some rolling/history info exists (so we are not
      training on "first-ever" matches for a team with no history).
    - Fill NaNs in features with 0.0.
    - Split into train / val by time.
    """
    df = df[~df[target_col].isna()].copy()

    # Any feature that contains "form" (multi-horizon team form) or starts with "h2h_"
    rolling_cols = [c for c in feature_cols if ("form" in c) or c.startswith("h2h_")]
    if rolling_cols:
        has_some_history = df[rolling_cols].notna().any(axis=1)
        df = df[has_some_history].copy()

    X = df[feature_cols].copy()
    y = df[target_col].astype(int).values

    # Simple NaN handling: treat missing as 0
    X = X.fillna(0.0)

    train_mask, val_mask = make_time_split(df)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    return X_train, X_val, y_train, y_val, feature_cols


def train_single_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_name: str,
):
    print(f"\n=== Training model for target: {target_col} ===")

    X_train, X_val, y_train, y_val, used_features = clean_training_data(df, feature_cols, target_col)

    print(f"Train size: {X_train.shape}, Validation size: {X_val.shape}")
    print(f"Positive rate train: {y_train.mean():.4f}, val: {y_val.mean():.4f}")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=50,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    val_proba = model.predict_proba(X_val)[:, 1]
    val_pred = (val_proba >= 0.5).astype(int)

    try:
        auc = roc_auc_score(y_val, val_proba)
    except ValueError:
        auc = float("nan")

    try:
        ap = average_precision_score(y_val, val_proba)
    except ValueError:
        ap = float("nan")

    print(f"AUC: {auc:.4f}, Average Precision (PR-AUC): {ap:.4f}")
    print("Classification report (threshold=0.5):")
    print(classification_report(y_val, val_pred, digits=3))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / f"{model_name}.joblib"
    metadata_path = MODELS_DIR / f"{model_name}_meta.json"

    joblib.dump(
        {
            "model": model,
            "features": used_features,
            "target": target_col,
        },
        model_path,
    )

    try:
        import json

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "target": target_col,
                    "model_file": str(model_path),
                    "n_features": len(used_features),
                },
                f,
                indent=2,
            )
    except Exception as e:
        print(f"Could not save metadata JSON: {e}")

    print(f"Saved model to {model_path}")
    print(f"Saved metadata to {metadata_path}")


def main():
    print("Loading feature dataset...")
    df = load_features()
    print(f"Features shape (full): {df.shape}")

    # -------------------------------------------
    # Global date filtering before splitting
    # -------------------------------------------
    if TRAIN_START_DATE is not None:
        start_dt = pd.to_datetime(TRAIN_START_DATE)
        df = df[df["match_date"] >= start_dt]

    if TRAIN_END_DATE is not None:
        end_dt = pd.to_datetime(TRAIN_END_DATE)
        df = df[df["match_date"] <= end_dt]

    print(f"Features shape after TRAIN window: {df.shape}")

    feature_cols = select_feature_columns(df)
    print(f"Using {len(feature_cols)} feature columns")

    # existing HT/FT pattern targets
    htft_targets = [
        ("target_ht_home_ft_home", "model_ht_home_ft_home"),
        ("target_ht_home_ft_draw", "model_ht_home_ft_draw"),
        ("target_ht_home_ft_away", "model_ht_home_ft_away"),
        ("target_ht_draw_ft_home", "model_ht_draw_ft_home"),
        ("target_ht_draw_ft_draw", "model_ht_draw_ft_draw"),
        ("target_ht_draw_ft_away", "model_ht_draw_ft_away"),
        ("target_ht_away_ft_home", "model_ht_away_ft_home"),
        ("target_ht_away_ft_draw", "model_ht_away_ft_draw"),
        ("target_ht_away_ft_away", "model_ht_away_ft_away"),
    ]

    # FT 1X2 binary targets
    ft_targets = [
        ("ft_home_win", "model_ft_home_win"),
        ("ft_draw", "model_ft_draw"),
        ("ft_away_win", "model_ft_away_win"),
    ]

    for target_col, model_name in htft_targets:
        if target_col not in df.columns:
            print(f"WARNING: missing {target_col}, skipping.")
            continue
        train_single_model(df, feature_cols, target_col, model_name)

    for target_col, model_name in ft_targets:
        if target_col not in df.columns:
            print(f"WARNING: missing {target_col}, skipping.")
            continue
        train_single_model(df, feature_cols, target_col, model_name)


if __name__ == "__main__":
    main()
