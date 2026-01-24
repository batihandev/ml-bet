import argparse
import pandas as pd
from typing import Optional
from pathlib import Path
from .engine import build_X, load_model, load_features


def get_predictions_df(pred_date: Optional[str] = None) -> pd.DataFrame:
    df = load_features()
    date_val = pd.to_datetime(pred_date).date() if pred_date else df["match_date"].max().date()
    df_day = df[df["match_date"].dt.date == date_val].copy()
    if df_day.empty: raise ValueError(f"No matches for date {date_val}")

    models = [
        ("ht_home_ft_home", "model_ht_home_ft_home"), ("ht_home_ft_draw", "model_ht_home_ft_draw"),
        ("ht_home_ft_away", "model_ht_home_ft_away"), ("ht_draw_ft_home", "model_ht_draw_ft_home"),
        ("ht_draw_ft_draw", "model_ht_draw_ft_draw"), ("ht_draw_ft_away", "model_ht_draw_ft_away"),
        ("ht_away_ft_home", "model_ht_away_ft_home"), ("ht_away_ft_draw", "model_ht_away_ft_draw"),
        ("ht_away_ft_away", "model_ht_away_ft_away"), ("ft_home_win", "model_ft_home_win"),
        ("ft_draw", "model_ft_draw"), ("ft_away_win", "model_ft_away_win"),
    ]

    for key, model_name in models:
        try:
            model, feats = load_model(model_name)
            X = build_X(df_day, feats)
            df_day[f"prob_{key}"] = model.predict_proba(X)[:, 1]
        except FileNotFoundError:
            continue
    return df_day

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None)
    args = parser.parse_args()
    df_day = get_predictions_df(args.date)
    print(df_day.head())

if __name__ == "__main__":
    main()
