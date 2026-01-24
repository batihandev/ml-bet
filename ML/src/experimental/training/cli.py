import os
import pandas as pd
from typing import Optional
from .data import load_features, select_feature_columns
from .engine import train_single_model

def run_training_process(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    cutoff_date: Optional[str] = None,
    n_estimators: int = 300,
    max_depth: int = 8,
    min_samples_leaf: int = 50,
):
    print("Loading feature dataset...")
    df = load_features()

    if start_date:
        df = df[df["match_date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["match_date"] <= pd.to_datetime(end_date)]

    feature_cols = select_feature_columns(df)
    
    targets = [
        ("target_ht_home_ft_home", "model_ht_home_ft_home"),
        ("target_ht_home_ft_draw", "model_ht_home_ft_draw"),
        ("target_ht_home_ft_away", "model_ht_home_ft_away"),
        ("target_ht_draw_ft_home", "model_ht_draw_ft_home"),
        ("target_ht_draw_ft_draw", "model_ht_draw_ft_draw"),
        ("target_ht_draw_ft_away", "model_ht_draw_ft_away"),
        ("target_ht_away_ft_home", "model_ht_away_ft_home"),
        ("target_ht_away_ft_draw", "model_ht_away_ft_draw"),
        ("target_ht_away_ft_away", "model_ht_away_ft_away"),
        ("ft_home_win", "model_ft_home_win"),
        ("ft_draw", "model_ft_draw"),
        ("ft_away_win", "model_ft_away_win"),
    ]

    for target_col, model_name in targets:
        if target_col in df.columns:
            train_single_model(
                df, feature_cols, target_col, model_name, cutoff_date,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf
            )

if __name__ == "__main__":
    run_training_process(
        start_date=os.getenv("TRAIN_START_DATE"),
        end_date=os.getenv("TRAIN_END_DATE"),
        cutoff_date=os.getenv("FIXED_CUTOFF_DATE")
    )
