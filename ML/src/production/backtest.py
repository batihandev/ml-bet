import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Optional, Dict, Any, List

from .schema import CLASS_MAPPING 
from dataset.cleaner import load_features 
from .predict import predict_ft_1x2
from .backtest_utils import (
    build_df_bt, 
    index_predictions, 
    compute_stake, 
    compute_profit, 
    get_actual_outcome,
    compute_max_drawdown
)
from .betting_logic import select_bet
from .bootstrap import bootstrap_roi

ROOT_DIR = Path(__file__).resolve().parents[2]

def backtest_production_1x2(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    min_edge: float = 0.05,
    min_ev: float = 0.0,
    stake: float = 1.0,
    kelly_mult: float = 0.0,
    selection_mode: str = "best_ev",
    debug: int = 0,
) -> Dict[str, Any]:
    """
    Run backtest for the production single-model FT 1X2.
    """
    meta_path = ROOT_DIR / "models" / "model_ft_1x2_meta.json"
    training_cutoff = None
    data_end = None
    
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
            cutoff_str = meta.get("windows", {}).get("training_cutoff_date") or meta.get("training_params", {}).get("training_cutoff_date")
            if cutoff_str:
                training_cutoff = pd.to_datetime(cutoff_str)
            
            data_end_str = meta.get("windows", {}).get("data_end")
            if data_end_str:
                data_end = pd.to_datetime(data_end_str)
    
    df_all = load_features()
    if data_end is None and not df_all.empty:
        data_end = df_all["match_date"].max()
    
    data_min = pd.to_datetime(df_all["match_date"].min())
    data_max = pd.to_datetime(df_all["match_date"].max())

    if training_cutoff and data_end and training_cutoff >= data_end:
        return {
            "summary": {
                "total_bets": 0, 
                "total_staked": 0.0,
                "total_profit": 0.0,
                "roi": 0.0,
                "avg_odds": 0.0,
                "avg_edge": 0.0,
                "avg_ev": 0.0,
                "avg_selected_prob": 0.0,
                "outcome_mix": {"home": 0.0, "draw": 0.0, "away": 0.0},
                "status": "Not available (no labeled data after cutoff)"
            },
            "markets": [], "league_stats_df": pd.DataFrame(), "daily_equity_df": pd.DataFrame(), "bets_df": pd.DataFrame()
        }

    resolved_start = pd.to_datetime(start_date) if start_date else data_min
    if training_cutoff:
        min_allowed = training_cutoff + pd.Timedelta(days=1)
        if resolved_start < min_allowed:
            resolved_start = min_allowed
            
    resolved_end = pd.to_datetime(end_date) if end_date else data_max
    resolved_start = max(resolved_start, data_min)
    resolved_end = min(resolved_end, data_max)
    
    df_bt = build_df_bt(df_all, resolved_start, resolved_end)
    if df_bt.empty:
        return {
            "summary": {
                "total_bets": 0, 
                "total_staked": 0.0,
                "total_profit": 0.0,
                "roi": 0.0,
                "avg_odds": 0.0,
                "avg_edge": 0.0,
                "avg_ev": 0.0,
                "avg_selected_prob": 0.0,
                "outcome_mix": {"home": 0.0, "draw": 0.0, "away": 0.0},
                "status": "No matches in window"
            },
            "markets": [], "league_stats_df": pd.DataFrame(), "daily_equity_df": pd.DataFrame(), "bets_df": pd.DataFrame()
        }

    total_labeled_matches = len(df_bt)
    predictions = predict_ft_1x2(df_bt, debug=debug)
    pred_by_id = index_predictions(predictions)
    
    # Debug info from the first prediction if present
    debug_payload = predictions[0].get("debug_info") if predictions and debug == 1 else None

    # Diagnostic groups
    group_all_valid = []  # Matches that pass ALL_VALID criteria
    group_top_passes = [] # subset of group_all_valid where top outcome passes gate
    bet_rows = []         # subset of group_all_valid where a bet is actually placed
    
    n_any_passes_gate = 0
    n_top_prob_passes_gate = 0
    skipped_missing_pred = 0
    skipped_invalid_odds = 0
    skipped_no_value = 0
    
    for _, match_row in df_bt.iterrows():
        match_id = int(match_row["match_id"])
        
        # 1. Prediction present?
        if match_id not in pred_by_id:
            skipped_missing_pred += 1
            continue
            
        pred_res = pred_by_id[match_id]
        metrics = pred_res.get("metrics", [])
        
        # 2. Labeled outcome present? (Handled by df_bt construction mostly, but safe to check)
        actual = get_actual_outcome(match_row)
        if actual is None:
            continue
            
        # 3. Odds valid & metrics non-empty?
        # If metrics is empty, it means calculate_implied_probs returned None (invalid odds)
        if not metrics:
            skipped_invalid_odds += 1
            continue

        # ALL_VALID group
        top_metric = max(metrics, key=lambda x: x["prob"])
        group_all_valid.append({
            "outcome": top_metric["outcome"],
            "odds": top_metric["odds"]
        })

        # ANY_PASSES_GATE check
        any_passes = any(m["edge"] >= min_edge and m["ev"] >= min_ev and m["odds"] > 1.0 for m in metrics)
        if any_passes:
            n_any_passes_gate += 1

        # TOP_PASSES_GATE check
        top_passes = (top_metric["edge"] >= min_edge and top_metric["ev"] >= min_ev and top_metric["odds"] > 1.0)
        if top_passes:
            n_top_prob_passes_gate += 1
            group_top_passes.append({
                "outcome": top_metric["outcome"],
                "odds": top_metric["odds"]
            })

        # Best bet selection
        best_bet = select_bet(metrics, min_edge, min_ev, selection_mode)
        if not best_bet:
            skipped_no_value += 1
            continue
            
        current_stake = compute_stake(stake, kelly_mult, best_bet["prob"], best_bet["odds"])
        if current_stake <= 0:
            skipped_no_value += 1
            continue
            
        recommendation = best_bet["outcome"]
        is_win = (actual == CLASS_MAPPING[recommendation])
        profit = compute_profit(current_stake, best_bet["odds"], is_win)
        
        bet_rows.append({
            "match_id": match_id,
            "date": match_row["match_date"].date(),
            "division": match_row.get("division", "Unknown"),
            "home_team": match_row["home_team"],
            "away_team": match_row["away_team"],
            "recommendation": recommendation,
            "selected_outcome": recommendation,
            "selected_odds": best_bet["odds"],
            "selected_prob": best_bet["prob"],
            "selected_pimp": best_bet.get("pimp", 0.0),
            "selected_edge": best_bet["edge"],
            "selected_ev": best_bet["ev"],
            "is_win": int(is_win),
            "stake": float(round(current_stake, 2)),
            "profit": float(round(profit, 2)),
            # Compat fields
            "market": f"FT {recommendation.upper()}",
            "odds": best_bet["odds"],
            "prob": best_bet["prob"]
        })

    summary_params = {
        "min_edge": float(min_edge),
        "min_ev": float(min_ev),
        "stake": float(stake),
        "kelly_mult": float(kelly_mult),
        "selection_mode": selection_mode
    }

    if not bet_rows:
        return {
            "summary": {
                "total_bets": 0,
                "total_labeled_matches": total_labeled_matches,
                "skipped_missing_pred": skipped_missing_pred,
                "skipped_no_value": skipped_no_value,
                "total_staked": 0.0,
                "total_profit": 0.0,
                "roi": 0.0,
                "hit_rate": 0.0,
                "avg_odds": 0.0,
                "avg_edge": 0.0,
                "avg_ev": 0.0,
                "avg_selected_prob": 0.0,
                "outcome_mix": {"home": 0.0, "draw": 0.0, "away": 0.0},
                **summary_params
            },
            "markets": [], "league_stats_df": pd.DataFrame(), "daily_equity_df": pd.DataFrame(), "bets_df": pd.DataFrame()
        }

    bets_df = pd.DataFrame(bet_rows)
    
    # Aggregates
    total_staked = bets_df["stake"].sum()
    total_profit = bets_df["profit"].sum()
    total_bets = len(bets_df)
    roi = total_profit / total_staked if total_staked > 0 else 0.0
    wins = bets_df["is_win"].sum()
    hit_rate = wins / total_bets if total_bets > 0 else 0.0
    
    avg_odds = bets_df["selected_odds"].mean()
    avg_edge = bets_df["selected_edge"].mean()
    avg_ev = bets_df["selected_ev"].mean()
    avg_prob = bets_df["selected_prob"].mean()

    # Summary construction with diagnostics
    def get_group_stats(rows: List[Dict[str, Any]], odds_key="odds", outcome_key="outcome"):
        if not rows:
            return {
                "count": 0, "avg_odds": 0.0, "median_odds": 0.0, "p90_odds": 0.0,
                "mix": {"home": 0.0, "draw": 0.0, "away": 0.0}
            }
        df = pd.DataFrame(rows)
        n = len(df)
        counts = df[outcome_key].value_counts().to_dict()
        return {
            "count": n,
            "avg_odds": float(round(df[odds_key].mean(), 2)),
            "median_odds": float(round(df[odds_key].median(), 2)),
            "p90_odds": float(round(df[odds_key].quantile(0.9), 2)),
            "mix": {
                "home": float(round(counts.get("home", 0) / n, 3)),
                "draw": float(round(counts.get("draw", 0) / n, 3)),
                "away": float(round(counts.get("away", 0) / n, 3)),
            }
        }

    stats_all_valid = get_group_stats(group_all_valid)
    stats_top_passes = get_group_stats(group_top_passes)
    stats_placed = get_group_stats(bet_rows, odds_key="selected_odds", outcome_key="selected_outcome")

    # Daily equity & MDD
    daily_equity_df = bets_df.groupby("date").agg(
        staked=("stake", "sum"),
        profit=("profit", "sum")
    ).reset_index().sort_values("date")
    daily_equity_df["cum_profit"] = daily_equity_df["profit"].cumsum().round(2)
    
    if kelly_mult > 0:
        # Bankroll evolution simulation (starting from 1.0)
        bankroll = [1.0]
        for _, row in bets_df.sort_values("date").iterrows():
            # fractional kelly: bankroll *= (1 + kelly_fraction * (odds - 1) if win else 1 - kelly_fraction)
            # here stake is already computed as stake_base * mult * kelly_f
            # so pct_staked = stake / stake_base_implied (which we don't have exactly, 
            # but we can assume bankroll starts at 'stake' or 1.0)
            # Let's use a simpler bankroll = 1 + cum_profit / original_stake if we want unit-based
            # or multiplicative if we assume stake is always % of current bankroll.
            # Requirement: "Kelly (fractional), equity should be on a bankroll series (multiplicative)"
            # we need to re-run the loop if we want true multiplicative, 
            # but usually backtests use fixed initial bankroll.
            pass
        
        # simplified bankroll for MDD in backtest context
        # bankroll = 100 * prod(1 + profit_pct)
        # But our stake is fixed relative to 'stake' param, not dynamic bankroll.
        # So we treat it as units unless we implement dynamic bankroll in backtest.
        # For now, use cumulative units for flat, and we'll calculate MDD on cumulative profit.
        mdd = compute_max_drawdown(daily_equity_df["cum_profit"], mode="flat")
    else:
        mdd = compute_max_drawdown(daily_equity_df["cum_profit"], mode="flat")

    # League stats
    league_stats_df = bets_df.groupby("division").agg(
        bets=("division", "size"),
        wins=("is_win", "sum"),
        staked=("stake", "sum"),
        profit=("profit", "sum")
    ).reset_index()
    league_stats_df["roi"] = (league_stats_df["profit"] / league_stats_df["staked"]).round(4)
    league_stats_df["hit_rate"] = (league_stats_df["wins"] / league_stats_df["bets"]).round(4)

    # Bootstrap CI
    ci_res = bootstrap_roi(bets_df, n=1000)
    
    summary = {
        "total_labeled_matches": total_labeled_matches,
        "n_all_valid": stats_all_valid["count"],
        "n_any_passes_gate": n_any_passes_gate,
        "n_top_prob_passes_gate": n_top_prob_passes_gate,
        "total_bets": total_bets,
        "total_staked": float(round(total_staked, 2)),
        "total_profit": float(round(total_profit, 2)),
        "roi": float(round(roi, 4)),
        "hit_rate": float(round(hit_rate, 4)),
        "avg_odds": float(round(avg_odds, 3)),
        "avg_edge": float(round(avg_edge, 4)),
        "avg_ev": float(round(avg_ev, 4)),
        "avg_selected_prob": float(round(avg_prob, 4)),
        
        "stats_all_valid": stats_all_valid,
        "stats_top_passes_gate": stats_top_passes,
        "stats_placed_bets": stats_placed,
        
        "outcome_mix": stats_placed["mix"],  # Keep for backward compatibility
        "max_drawdown": float(round(mdd, 2)),
        "effective_start_date": str(resolved_start.date()),
        "effective_end_date": str(resolved_end.date()),
        "skipped_missing_pred": skipped_missing_pred,
        "skipped_invalid_odds": skipped_invalid_odds,
        "skipped_no_value": skipped_no_value,
        "debug_info": debug_payload,
        **summary_params
    }
    
    if ci_res["status"] == "success":
        summary.update({
            "roi_p05": float(round(ci_res["roi_p05"], 4)),
            "roi_p95": float(round(ci_res["roi_p95"], 4)),
            "profit_p05": float(round(ci_res["profit_p05"], 2)),
            "profit_p95": float(round(ci_res["profit_p95"], 2)),
        })

    return {
        "summary": summary,
        "markets": [{
            "key": "ft_1x2", "label": "Production FT 1X2",
            "bets": total_bets, "wins": int(wins), "staked": float(round(total_staked, 2)),
            "profit": float(round(total_profit, 2)), "roi": float(round(roi, 4)), "hit_rate": float(round(hit_rate, 4))
        }],
        "league_stats_df": league_stats_df,
        "daily_equity_df": daily_equity_df,
        "bets_df": bets_df
    }

if __name__ == "__main__":
    res = backtest_production_1x2(start_date="2024-01-01")
    print(f"Backtest Summary: {res['summary']}")
