import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Optional, Dict, Any, List
from .predict import predict_ft_1x2
from .schema import CLASS_MAPPING, TARGET_COL
from dataset.cleaner import load_features

ROOT_DIR = Path(__file__).resolve().parents[2]

def backtest_production_1x2(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    min_edge: float = 0.05,
    stake: float = 1.0,
    kelly_mult: float = 0.0,
) -> Dict[str, Any]:
    """
    Run backtest for the production single-model FT 1X2.
    """
    # Check model meta for training cutoff
    meta_path = ROOT_DIR / "models" / "model_ft_1x2_meta.json"
    training_cutoff = None
    data_end = pd.to_datetime("2025-06-30")
    
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
            cutoff_str = meta.get("training_params", {}).get("training_cutoff_date")
            if cutoff_str:
                training_cutoff = pd.to_datetime(cutoff_str)
    
    if training_cutoff and training_cutoff >= data_end:
        return {
            "summary": {"total_bets": 0, "status": "Not available (no labeled data after cutoff)"},
            "markets": [],
            "league_stats_df": pd.DataFrame(),
            "daily_equity_df": pd.DataFrame(),
            "bets_df": pd.DataFrame()
        }

    df_all = load_features()
    
    # Resolve and clamp start_date
    resolved_start = pd.to_datetime(start_date) if start_date else df_all["match_date"].min()
    if training_cutoff:
        min_allowed = training_cutoff + pd.Timedelta(days=1)
        if resolved_start < min_allowed:
            print(f"Clamping backtest start from {resolved_start.date()} to {min_allowed.date()} (Training cutoff + 1 day)")
            resolved_start = min_allowed
    
    df_all = df_all[df_all["match_date"] >= resolved_start]
    if end_date:
        df_all = df_all[df_all["match_date"] <= pd.to_datetime(end_date)]

    if df_all.empty:
        return {
            "summary": {"total_bets": 0},
            "markets": [],
            "league_stats_df": pd.DataFrame(),
            "daily_equity_df": pd.DataFrame(),
            "bets_df": pd.DataFrame()
        }

    # Prepare actual outcomes for evaluation
    def get_actual_outcome(row):
        h = row.get("ft_home_goals")
        a = row.get("ft_away_goals")
        if pd.isna(h) or pd.isna(a):
            return None
        if h > a: return CLASS_MAPPING["home"]
        if h < a: return CLASS_MAPPING["away"]
        return CLASS_MAPPING["draw"]

    # Generate predictions
    predictions = predict_ft_1x2(df_all)
    
    bet_rows = []
    
    for i, res in enumerate(predictions):
        match_row = df_all.iloc[i]
        actual = get_actual_outcome(match_row)
        if actual is None:
            continue
            
        # We look at all 3 outcomes to see if any meet the min_edge
        # In experimental, it was separate models. Here it's 1X2.
        # We pick the outcome with the highest EV that meets min_edge.
        valid_bets = [
            m for m in res["metrics"] 
            if m["edge"] >= min_edge and m["odds"] > 1.0
        ]
        
        if not valid_bets:
            continue
            
        # Pick best value
        best_bet = max(valid_bets, key=lambda x: x["ev"])
        recommendation = best_bet["outcome"]
        prob = best_bet["prob"]
        odds = best_bet["odds"]
        
        # Determine stake
        if kelly_mult > 0:
            kelly_f = (prob * odds - 1.0) / (odds - 1.0)
            if kelly_f <= 0:
                continue
            # Align with experimental: stake * kelly_mult * f_kelly
            # But experimental doesn't usually use a '100 units' base bankroll.
            # However, for UI clarity, we'll stick to 'units' where stake is a bankroll multiplier.
            # If user sets stake=100 and mult=0.25, bet is 25% of 100 * kelly_f.
            # If user sets stake=1.0 and mult=0.25, let's assume stake means bankroll.
            current_stake = stake * kelly_mult * kelly_f * 100
        else:
            current_stake = stake
            
        if current_stake <= 0:
            continue
            
        is_win = (actual == CLASS_MAPPING[recommendation])
        profit = current_stake * (odds - 1.0) if is_win else -current_stake
        
        bet_rows.append({
            "date": match_row["match_date"].date(),
            "division": match_row.get("division", "Unknown"),
            "home_team": match_row["home_team"],
            "away_team": match_row["away_team"],
            "market": f"FT {recommendation.upper()}",
            "recommendation": recommendation,
            "odds": odds,
            "prob": prob,
            "is_win": int(is_win),
            "stake": float(round(current_stake, 2)),
            "profit": float(round(profit, 2))
        })

    if not bet_rows:
        return {
            "summary": {"total_bets": 0},
            "markets": [],
            "league_stats_df": pd.DataFrame(),
            "daily_equity_df": pd.DataFrame(),
            "bets_df": pd.DataFrame()
        }

    bets_df = pd.DataFrame(bet_rows)
    
    # Generate summary stats
    total_staked = bets_df["stake"].sum()
    total_profit = bets_df["profit"].sum()
    total_bets = len(bets_df)
    roi = total_profit / total_staked if total_staked > 0 else 0.0
    wins = bets_df["is_win"].sum()
    hit_rate = wins / total_bets if total_bets > 0 else 0.0

    # League stats
    league_stats_df = bets_df.groupby("division").agg(
        bets=("division", "size"),
        wins=("is_win", "sum"),
        staked=("stake", "sum"),
        profit=("profit", "sum")
    ).reset_index()
    league_stats_df["roi"] = (league_stats_df["profit"] / league_stats_df["staked"]).round(4)
    league_stats_df["hit_rate"] = (league_stats_df["wins"] / league_stats_df["bets"]).round(4)
    league_stats_df["profit"] = league_stats_df["profit"].round(2)

    # Daily equity
    daily_equity_df = bets_df.groupby("date").agg(
        staked=("stake", "sum"),
        profit=("profit", "sum")
    ).reset_index().sort_values("date")
    daily_equity_df["cum_profit"] = daily_equity_df["profit"].cumsum().round(2)
    daily_equity_df["profit"] = daily_equity_df["profit"].round(2)
    daily_equity_df["date"] = daily_equity_df["date"].astype(str)

    # Markets (just 1X2 for production)
    markets_summary = [{
        "key": "ft_1x2",
        "label": "Production FT 1X2",
        "bets": total_bets,
        "wins": int(wins),
        "staked": float(round(total_staked, 2)),
        "profit": float(round(total_profit, 2)),
        "roi": float(round(roi, 4)),
        "hit_rate": float(round(hit_rate, 4))
    }]

    return {
        "summary": {
            "total_matches": len(df_all),
            "total_bets": total_bets,
            "total_staked": float(round(total_staked, 2)),
            "total_profit": float(round(total_profit, 2)),
            "roi": float(round(roi, 4)),
            "hit_rate": float(round(hit_rate, 4)),
            "start_date": str(start_date),
            "end_date": str(end_date)
        },
        "markets": markets_summary,
        "league_stats_df": league_stats_df,
        "daily_equity_df": daily_equity_df,
        "bets_df": bets_df
    }

if __name__ == "__main__":
    res = backtest_production_1x2(start_date="2024-01-01")
    print(f"Backtest Summary: {res['summary']}")
