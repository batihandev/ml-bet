import numpy as np
import pandas as pd
from .base import ensure_match_datetime, infer_season_year

def add_attack_strength(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    if "season" not in df.columns:
        df["season"] = infer_season_year(df["match_date"])

    df["ft_home_goals"] = pd.to_numeric(df["ft_home_goals"], errors="coerce")
    df["ft_away_goals"] = pd.to_numeric(df["ft_away_goals"], errors="coerce")

    if "match_id" not in df.columns:
        df["match_id"] = range(len(df))
    df = ensure_match_datetime(df)
    sort_cols = ["division", "season", "match_datetime", "match_date", "match_id"] if "match_datetime" in df.columns else ["division", "season", "match_date", "match_id"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    df["league_avg_home_goals_prev"] = (
        df.groupby(["division", "season"])["ft_home_goals"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=[0, 1], drop=True)
    )
    df["league_avg_away_goals_prev"] = (
        df.groupby(["division", "season"])["ft_away_goals"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=[0, 1], drop=True)
    )

    df["home_scored_prev_avg"] = (
        df.groupby(["division", "season", "home_team"])["ft_home_goals"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=[0, 1, 2], drop=True)
    )
    df["away_scored_prev_avg"] = (
        df.groupby(["division", "season", "away_team"])["ft_away_goals"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=[0, 1, 2], drop=True)
    )

    df["home_attack_strength"] = df["home_scored_prev_avg"] / df["league_avg_home_goals_prev"].where(
        df["league_avg_home_goals_prev"] > 0
    )
    df["away_attack_strength"] = df["away_scored_prev_avg"] / df["league_avg_away_goals_prev"].where(
        df["league_avg_away_goals_prev"] > 0
    )

    df = df.drop(
        columns=[
            "league_avg_home_goals_prev",
            "league_avg_away_goals_prev",
            "home_scored_prev_avg",
            "away_scored_prev_avg",
        ],
        errors="ignore",
    )
    return df

def add_rule_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rule_score_home_htdftw"] = df[
        ["home_form5_htd_ftw_rate", "away_form5_htd_ftw_rate", "h2h_htd_ft_home_win_rate"]
    ].mean(axis=1)

    df["rule_score_away_htdftw"] = df[
        ["home_form5_htd_ftw_rate", "away_form5_htd_ftw_rate", "h2h_htd_ft_away_win_rate"]
    ].mean(axis=1)

    df["passes_rule_home_htdftw"] = (
        (df["home_form5_htd_ftw_rate"] >= 0.51)
        & (df["away_form5_htd_ftw_rate"] >= 0.51)
        & (df["h2h_htd_ft_home_win_rate"] >= 0.51)
    ).astype(int)

    df["passes_rule_away_htdftw"] = (
        (df["home_form5_htd_ftw_rate"] >= 0.51)
        & (df["away_form5_htd_ftw_rate"] >= 0.51)
        & (df["h2h_htd_ft_away_win_rate"] >= 0.51)
    ).astype(int)
    return df

def add_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"odd_home", "odd_draw", "odd_away"}.issubset(df.columns):
        df["inv_home"] = 1.0 / df["odd_home"].where(df["odd_home"] > 1.0)
        df["inv_draw"] = 1.0 / df["odd_draw"].where(df["odd_draw"] > 1.0)
        df["inv_away"] = 1.0 / df["odd_away"].where(df["odd_away"] > 1.0)

        df["inv_total"] = (
            df["inv_home"].fillna(0.0)
            + df["inv_draw"].fillna(0.0)
            + df["inv_away"].fillna(0.0)
        )

        df["p_home_imp_norm"] = df["inv_home"] / df["inv_total"].where(df["inv_total"] > 0)
        df["p_draw_imp_norm"] = df["inv_draw"] / df["inv_total"].where(df["inv_total"] > 0)
        df["p_away_imp_norm"] = df["inv_away"] / df["inv_total"].where(df["inv_total"] > 0)

        df["fav_prob_norm"] = df[["p_home_imp_norm", "p_away_imp_norm"]].max(axis=1)
        df["market_balance"] = (df["p_home_imp_norm"] - df["p_away_imp_norm"]).abs()
    else:
        for col in ["inv_home", "inv_draw", "inv_away", "inv_total", "p_home_imp_norm", "p_draw_imp_norm", "p_away_imp_norm", "fav_prob_norm", "market_balance"]:
            df[col] = np.nan
    return df
