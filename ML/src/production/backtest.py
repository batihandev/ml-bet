import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Optional, Dict, Any, List
# Fix schema.py to remove TARGET_COL if it's not needed or ensure it's imported correctly
from .schema import CLASS_MAPPING 
from dataset.cleaner import load_features 
from .utils_market import is_valid_odds
from .predict import predict_ft_1x2 # Relative import works if run as module

ROOT_DIR = Path(__file__).resolve().parents[2]

def backtest_production_1x2(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    min_edge: float = 0.05,
    min_ev: float = 0.0,
    stake: float = 1.0,
    kelly_mult: float = 0.0,
) -> Dict[str, Any]:
    """
    Run backtest for the production single-model FT 1X2.
    """
    # Check model meta for training cutoff
    meta_path = ROOT_DIR / "models" / "model_ft_1x2_meta.json"
    training_cutoff = None
    data_end = None
    
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
            # 5. Canonical meta field for cutoff
            cutoff_str = meta.get("windows", {}).get("training_cutoff_date")
            if not cutoff_str:
                cutoff_str = meta.get("training_params", {}).get("training_cutoff_date")
            
            if cutoff_str:
                training_cutoff = pd.to_datetime(cutoff_str)
            
            # 4. Remove hardcoded data_end
            data_end_str = meta.get("windows", {}).get("data_end")
            if data_end_str:
                data_end = pd.to_datetime(data_end_str)
    
    # Load all data
    df_all = load_features()
    
    # 4. Fallback data_end from data
    if data_end is None and not df_all.empty:
        data_end = df_all["match_date"].max()
    
    # Capture available data range for clamping
    data_min = pd.to_datetime(df_all["match_date"].min())
    data_max = pd.to_datetime(df_all["match_date"].max())

    # Early exit if cutoff is in future of data/logic invalid
    if training_cutoff and data_end and training_cutoff >= data_end:
        return {
            "summary": {"total_bets": 0, "status": "Not available (no labeled data after cutoff)"},
            "markets": [], "league_stats_df": pd.DataFrame(), "daily_equity_df": pd.DataFrame(), "bets_df": pd.DataFrame()
        }

    # 3. Date Resolution
    resolved_start = pd.to_datetime(start_date) if start_date else data_min
    
    if training_cutoff:
        min_allowed = training_cutoff + pd.Timedelta(days=1)
        if resolved_start < min_allowed:
            print(f"Clamping backtest start from {resolved_start.date()} to {min_allowed.date()} (Training cutoff + 1 day)")
            resolved_start = min_allowed
            
    resolved_end = pd.to_datetime(end_date) if end_date else data_max
    
    # Clamp to actual data availability
    resolved_start = max(resolved_start, data_min)
    resolved_end = min(resolved_end, data_max)
    
    # Filter by date
    df_window = df_all[
        (df_all["match_date"] >= resolved_start) & 
        (df_all["match_date"] <= resolved_end)
    ].copy()

    total_matches_in_window = len(df_window)

    if df_window.empty:
        return {
            "summary": {"total_bets": 0, "status": "No matches in window"},
            "markets": [], "league_stats_df": pd.DataFrame(), "daily_equity_df": pd.DataFrame(), "bets_df": pd.DataFrame()
        }

    # 2. Filter to labeled rows before predicting
    # We only care about rows we can evaluate (labeled)
    df_bt = df_window[
        df_window["ft_home_goals"].notna() & 
        df_window["ft_away_goals"].notna()
    ].copy()
    
    total_labeled_matches = len(df_bt)
    skipped_unlabeled = total_matches_in_window - total_labeled_matches

    if df_bt.empty:
         return {
            "summary": {"total_bets": 0, "skipped_unlabeled": skipped_unlabeled},
            "markets": [], "league_stats_df": pd.DataFrame(), "daily_equity_df": pd.DataFrame(), "bets_df": pd.DataFrame()
        }

    # 1. Normalize match_id to ensure consistent types
    df_bt["match_id"] = df_bt["match_id"].astype("int64")

    # 3. Guard against duplicate match_id
    if df_bt["match_id"].duplicated().any():
        raise ValueError("Duplicate match_id in backtest window")

    # Prepare actual outcomes for evaluation
    def get_actual_outcome(row):
        h = row.get("ft_home_goals")
        a = row.get("ft_away_goals")
        if pd.isna(h) or pd.isna(a):
            return None
        if h > a: return CLASS_MAPPING["home"]
        if h < a: return CLASS_MAPPING["away"]
        return CLASS_MAPPING["draw"]

    # Generate predictions on filtered set
    predictions = predict_ft_1x2(df_bt)
    
    # 8. Sanity check prediction count
    if len(predictions) != len(df_bt):
        print(f"Warning: predictions={len(predictions)} != rows={len(df_bt)}")
    
    # 4. Make pred_by_id robust
    pred_by_id = {}
    for r in predictions:
        mid = int(r["match_id"])
        if mid in pred_by_id:
            raise ValueError(f"Duplicate prediction for match_id={mid}")
        pred_by_id[mid] = r
    
    bet_rows = []
    
    # Counters
    skipped_missing_pred = 0
    skipped_invalid_odds = 0
    skipped_no_value = 0
    
    for idx, match_row in df_bt.iterrows():
        match_id = int(match_row["match_id"])
        
        # Alignment check
        if match_id not in pred_by_id:
            skipped_missing_pred += 1
            continue
            
        pred_res = pred_by_id[match_id]
        metrics = pred_res.get("metrics", [])
        
        # 5. Simplify odds validity logic
        # Treat empty metrics as invalid odds (predict.py centralizes this check)
        if not metrics:
            skipped_invalid_odds += 1
            continue
            
        # Optional double-check if desired, but metrics-gating handles it cleaner
        # We rely on predict() setting metrics=[] when odds are invalid.
            
        actual = get_actual_outcome(match_row)
        if actual is None:
            continue
            
        # 9. Bet Decision Logic (Centralized here)
        # 7. Enforce EV threshold in betting filter
        valid_bets = [
            m for m in metrics 
            if m["edge"] >= min_edge and m["ev"] >= min_ev and m["odds"] > 1.0
        ]
        
        if not valid_bets:
            skipped_no_value += 1
            continue
            
        # Pick best value
        best_bet = max(valid_bets, key=lambda x: x["ev"])
        recommendation = best_bet["outcome"]
        prob = best_bet["prob"]
        odds = best_bet["odds"]
        
        # 6. Fix Kelly stake sizing
        if kelly_mult > 0:
            kelly_f = (prob * odds - 1.0) / (odds - 1.0)
            if kelly_f <= 0:
                skipped_no_value += 1
                continue
            
            # Pure kelly stake
            current_stake = stake * kelly_mult * kelly_f
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

    # 6. Include all decision params in summary
    summary_params = {
        "min_edge": float(min_edge),
        "min_ev": float(min_ev),
        "stake": float(stake),
        "kelly_mult": float(kelly_mult)
    }

    if not bet_rows:
        return {
            "summary": {
                "total_bets": 0,
                "total_matches_in_window": total_matches_in_window,
                "total_labeled_matches": total_labeled_matches,
                "skipped_unlabeled": skipped_unlabeled,
                "skipped_missing_pred": skipped_missing_pred,
                "skipped_invalid_odds": skipped_invalid_odds,
                "skipped_no_value": skipped_no_value,
                **summary_params
            },
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
            "total_matches_in_window": total_matches_in_window,
            "total_labeled_matches": total_labeled_matches,
            "total_bets": total_bets,
            "total_staked": float(round(total_staked, 2)),
            "total_profit": float(round(total_profit, 2)),
            "roi": float(round(roi, 4)),
            "hit_rate": float(round(hit_rate, 4)),
            "requested_start_date": str(start_date),
            "requested_end_date": str(end_date),
            "effective_start_date": str(resolved_start.date()),
            "effective_end_date": str(resolved_end.date()),
            "skipped_unlabeled": skipped_unlabeled,
            "skipped_missing_pred": skipped_missing_pred,
            "skipped_invalid_odds": skipped_invalid_odds,
            "skipped_no_value": skipped_no_value,
            **summary_params
        },
        "markets": markets_summary,
        "league_stats_df": league_stats_df,
        "daily_equity_df": daily_equity_df,
        "bets_df": bets_df
    }

if __name__ == "__main__":
    res = backtest_production_1x2(start_date="2024-01-01")
    print(f"Backtest Summary: {res['summary']}")
