from datetime import date
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd
import warnings

from config.allowed_divisions import ALLOWED_DIVISIONS
from prediction.engine import load_features, load_model, build_X

# Suppress warnings
warnings.filterwarnings(
    "ignore",
    message="`sklearn.utils.parallel.delayed` should be used with "
            "`sklearn.utils.parallel.Parallel` to make it possible to "
            "propagate the scikit-learn configuration of the current "
            "thread to the joblib workers.",
    category=UserWarning,
)

ROOT_DIR = Path(__file__).resolve().parents[2]

def load_all_dates(df_all: pd.DataFrame) -> list[date]:
    return sorted(df_all["match_date"].dt.date.unique())

def _empty_result(
    start_date: Optional[str],
    end_date: Optional[str],
    min_edge: float,
    stake: float,
    kelly_mult: float,
    rows_before: int,
    rows_after: int,
) -> dict:
    return {
        "summary": {
            "start_date": start_date,
            "end_date": end_date,
            "min_edge": float(min_edge),
            "stake": float(stake),
            "kelly_mult": float(kelly_mult),
            "rows_before_division_filter": int(rows_before),
            "rows_after_division_filter": int(rows_after),
            "total_matches": 0,
            "total_bets": 0,
            "total_staked": 0.0,
            "total_profit": 0.0,
            "overall_roi": 0.0,
        },
        "markets": [],
        "league_stats_df": pd.DataFrame(),
        "daily_equity_df": pd.DataFrame(),
        "bets_df": pd.DataFrame(),
    }

def backtest_ft_1x2_core(
    start_date: Optional[str],
    end_date: Optional[str],
    min_edge: float,
    stake: float,
    kelly_mult: float = 0.0,
) -> dict:
    df_all = load_features()

    if "division" not in df_all.columns:
        raise ValueError("Column 'division' not found in features.csv")

    rows_before = len(df_all)
    df_all = df_all[df_all["division"].isin(ALLOWED_DIVISIONS)].copy()
    rows_after = len(df_all)

    if {"odd_home", "odd_draw", "odd_away"}.issubset(df_all.columns):
        df_all["inv_home"] = 1.0 / df_all["odd_home"].where(df_all["odd_home"] > 1.0)
        df_all["inv_draw"] = 1.0 / df_all["odd_draw"].where(df_all["odd_draw"] > 1.0)
        df_all["inv_away"] = 1.0 / df_all["odd_away"].where(df_all["odd_away"] > 1.0)
        df_all["inv_total"] = (
            df_all["inv_home"].fillna(0.0)
            + df_all["inv_draw"].fillna(0.0)
            + df_all["inv_away"].fillna(0.0)
        )
    else:
        df_all["inv_home"] = np.nan
        df_all["inv_draw"] = np.nan
        df_all["inv_away"] = np.nan
        df_all["inv_total"] = np.nan

    all_dates = load_all_dates(df_all)
    if not all_dates:
        return _empty_result(start_date, end_date, min_edge, stake, kelly_mult, rows_before, rows_after)

    if start_date is None:
        sd = all_dates[0]
    else:
        sd = pd.to_datetime(start_date).date()

    if end_date is None:
        ed = all_dates[-1]
    else:
        ed = pd.to_datetime(end_date).date()

    date_list = [d for d in all_dates if sd <= d <= ed]
    if not date_list:
        return _empty_result(str(sd), str(ed), min_edge, stake, kelly_mult, rows_before, rows_after)

    markets = [
        {"key": "ft_home_win", "label": "FT HOME (1)", "model_name": "model_ft_home_win", "odds_col": "odd_home", "target_col": "ft_home_win", "inv_col": "inv_home"},
        {"key": "ft_draw", "label": "FT DRAW (X)", "model_name": "model_ft_draw", "odds_col": "odd_draw", "target_col": "ft_draw", "inv_col": "inv_draw"},
        {"key": "ft_away_win", "label": "FT AWAY (2)", "model_name": "model_ft_away_win", "odds_col": "odd_away", "target_col": "ft_away_win", "inv_col": "inv_away"},
    ]

    models: dict[str, dict] = {}
    for m in markets:
        try:
            model, feats = load_model(m["model_name"])
            models[m["key"]] = {**m, "model": model, "features": feats}
        except FileNotFoundError:
            continue

    if not models:
        return _empty_result(str(sd), str(ed), min_edge, stake, kelly_mult, rows_before, rows_after)

    total_matches = 0
    total_bets = 0
    total_staked = 0.0
    total_profit = 0.0
    bet_rows = []

    bets_count = {k: 0 for k in models.keys()}
    wins_count = {k: 0 for k in models.keys()}
    stake_sum = {k: 0.0 for k in models.keys()}
    profit_sum = {k: 0.0 for k in models.keys()}

    for d in date_list:
        mask_day = df_all["match_date"].dt.date == d
        df_day = df_all[mask_day].copy()
        if df_day.empty: continue

        total_matches += len(df_day)

        for key, info in models.items():
            if info["odds_col"] not in df_day.columns or info["target_col"] not in df_day.columns:
                continue

            X = build_X(df_day, info["features"])
            p_raw_all = info["model"].predict_proba(X)[:, 1]
            df_day[f"prob_raw_{key}"] = p_raw_all

            for _, row in df_day.iterrows():
                p_raw = row[f"prob_raw_{key}"]
                odds = row[info["odds_col"]]
                outcome = row[info["target_col"]]

                if pd.isna(p_raw) or pd.isna(odds) or odds <= 1.0:
                    continue

                p_hat = float(np.clip(p_raw, 0.001, 0.999))
                inv_total = row.get("inv_total", np.nan)
                p_imp = float(row.get(info["inv_col"], np.nan) / inv_total) if not pd.isna(inv_total) and inv_total > 0 else 1.0 / odds

                edge = p_hat - p_imp
                if edge < min_edge: continue

                if kelly_mult > 0.0:
                    f_kelly = (p_hat * odds - 1.0) / (odds - 1.0)
                    if f_kelly <= 0: continue
                    bet_size = stake * kelly_mult * f_kelly
                else:
                    bet_size = stake

                bets_count[key] += 1
                total_bets += 1
                stake_sum[key] += bet_size
                total_staked += bet_size

                profit = bet_size * (odds - 1.0) if outcome == 1 else -bet_size
                profit_sum[key] += profit
                total_profit += profit
                if outcome == 1: wins_count[key] += 1

                bet_rows.append({
                    "date": d, "division": row.get("division"), "home_team": row.get("home_team"),
                    "away_team": row.get("away_team"), "market": info["label"], "model_key": key,
                    "odds": float(odds), "prob_raw": float(p_raw), "prob_cal": float(p_hat),
                    "prob_implied": float(p_imp), "edge": float(edge), "stake": float(bet_size),
                    "profit": float(profit), "is_win": int(outcome == 1),
                })

    if total_bets == 0:
        return _empty_result(str(sd), str(ed), min_edge, stake, kelly_mult, rows_before, rows_after)

    bets_df = pd.DataFrame(bet_rows)
    league_stats_df = bets_df.groupby("division").agg(bets=("division","size"), wins=("is_win","sum"), staked=("stake","sum"), profit=("profit","sum")).reset_index()
    league_stats_df["hit_rate"] = league_stats_df["wins"] / league_stats_df["bets"]
    league_stats_df["roi"] = league_stats_df["profit"] / league_stats_df["staked"]

    daily_equity_df = bets_df.groupby("date").agg(staked=("stake","sum"), profit=("profit","sum")).reset_index().sort_values("date")
    daily_equity_df["cum_profit"] = daily_equity_df["profit"].cumsum()
    daily_equity_df["cum_staked"] = daily_equity_df["staked"].cumsum()
    daily_equity_df["cum_roi"] = daily_equity_df["cum_profit"] / daily_equity_df["cum_staked"]

    overall_roi = total_profit / total_staked if total_staked > 0 else 0.0
    markets_summary = [{
        "key": k, "label": models[k]["label"], "bets": int(bets_count[k]), "wins": int(wins_count[k]),
        "staked": float(stake_sum[k]), "profit": float(profit_sum[k]),
        "roi": float(profit_sum[k]/stake_sum[k] if stake_sum[k]>0 else 0.0),
        "hit_rate": float(wins_count[k]/bets_count[k] if bets_count[k]>0 else 0.0)
    } for k in models]

    return {
        "summary": {
            "start_date": str(sd), "end_date": str(ed), "min_edge": float(min_edge),
            "stake": float(stake), "kelly_mult": float(kelly_mult),
            "rows_before_division_filter": int(rows_before), "rows_after_division_filter": int(rows_after),
            "total_matches": int(total_matches), "total_bets": int(total_bets),
            "total_staked": float(total_staked), "total_profit": float(total_profit),
            "overall_roi": float(overall_roi),
        },
        "markets": markets_summary,
        "league_stats_df": league_stats_df,
        "daily_equity_df": daily_equity_df,
        "bets_df": bets_df,
    }
