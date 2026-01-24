import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Callable
from .backtest_utils import (
    build_df_bt, index_predictions, 
    compute_stake, compute_profit, get_actual_outcome, compute_kelly_fraction
)
from .betting_logic import select_bet, metric_passes_gate
from .predict import predict_ft_1x2
from .bootstrap import bootstrap_roi
from dataset.cleaner import load_features

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
LATEST_SWEEP_PATH = DATA_DIR / "backtest_sweep_latest.json"

def save_latest_sweep(data: Dict[str, Any]):
    """Save the sweep results to the data folder."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(LATEST_SWEEP_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save sweep results: {e}")

def load_latest_sweep() -> Optional[Dict[str, Any]]:
    """Load the latest sweep results from the data folder."""
    if not LATEST_SWEEP_PATH.exists():
        return None
    try:
        with open(LATEST_SWEEP_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load sweep results: {e}")
        return None

def run_backtest_sweep(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    edge_range: Tuple[float, float, float] = (0.0, 0.10, 0.01),
    ev_range: Tuple[float, float, float] = (0.0, 0.10, 0.01),
    stake: float = 1.0,
    kelly_mult: float = 0.0,
    min_bets: int = 300,
    bootstrap_n: int = 500,
    max_ci_cells: int = 50,
    selection_mode: str = "best_ev",
    debug: int = 0,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    progress_every: int = 50
) -> Dict[str, Any]:
    """
    Perform an optimized grid search over (min_edge, min_ev).
    Predicts once and reuses metrics.
    """
    df_all = load_features()
    
    # Resolve dates
    data_min = pd.to_datetime(df_all["match_date"].min())
    data_max = pd.to_datetime(df_all["match_date"].max())
    resolved_start = pd.to_datetime(start_date) if start_date else data_min
    resolved_end = pd.to_datetime(end_date) if end_date else data_max
    
    df_bt = build_df_bt(df_all, resolved_start, resolved_end)
    if df_bt.empty:
        return {"cells": [], "status": "No matches in window"}

    # Predict once
    predictions = predict_ft_1x2(df_bt, debug=0) # sweep usually doesn't need debug rows per cell
    pred_by_id = index_predictions(predictions)
    
    # Prepare all possible matches info
    # ALL_VALID: pred present + actual present + valid odds + non-empty metrics
    processed_matches = []
    n_total_all_valid = 0
    
    for _, match_row in df_bt.iterrows():
        mid = int(match_row["match_id"])
        if mid not in pred_by_id: continue
        actual = get_actual_outcome(match_row)
        if actual is None: continue
        
        metrics = pred_by_id[mid]["metrics"]
        if not metrics: continue # Invalid odds
        
        # Determine top-prob outcome once
        top_metric = max(metrics, key=lambda x: x["prob"])
        
        processed_matches.append({
            "id": mid,
            "date": match_row["match_date"],
            "actual": actual,
            "metrics": metrics,
            "top_outcome": top_metric["outcome"],
            "top_odds": top_metric["odds"],
            "top_prob": top_metric["prob"],
            "top_edge": top_metric["edge"],
            "top_ev": top_metric["ev"]
        })
        n_total_all_valid += 1
    
    # Generate grid
    edges = np.round(np.arange(edge_range[0], edge_range[1] + 0.0001, edge_range[2]), 3)
    evs = np.round(np.arange(ev_range[0], ev_range[1] + 0.0001, ev_range[2]), 3)
    total_cells = int(len(edges) * len(evs))
    cell_idx = 0
    
    cells = []
    
    for me in edges:
        for mv in evs:
            cell_idx += 1
            bet_rows = []
            n_any_passes_gate = 0
            n_top_prob_passes_gate = 0
            
            # For distributions
            group_top_prob_all_valid = []
            group_top_prob_passes_gate = []

            bankroll = float(stake) if kelly_mult > 0 else None
            
            for m in processed_matches:
                # ALL_VALID is constant for all cells
                group_top_prob_all_valid.append({
                    "outcome": m["top_outcome"],
                    "odds": m["top_odds"]
                })
                
                # Check gates
                any_passes = any(metric_passes_gate(met, me, mv, selection_mode) for met in m["metrics"])
                if any_passes:
                    n_any_passes_gate += 1

                top_passes = metric_passes_gate(
                    {"edge": m["top_edge"], "ev": m["top_ev"], "odds": m["top_odds"]},
                    me,
                    mv,
                    selection_mode
                )
                if top_passes:
                    n_top_prob_passes_gate += 1
                    group_top_prob_passes_gate.append({
                        "outcome": m["top_outcome"],
                        "odds": m["top_odds"]
                    })
                
                # Selection logic
                best = select_bet(m["metrics"], me, mv, selection_mode)
                if not best: continue
                
                from .schema import CLASS_MAPPING
                is_win = (m["actual"] == CLASS_MAPPING[best["outcome"]])

                if kelly_mult > 0:
                    if bankroll is None or bankroll <= 0:
                        continue
                    kelly_f = compute_kelly_fraction(best["prob"], best["odds"])
                    s = bankroll * kelly_mult * kelly_f
                    if s <= 0:
                        continue
                    if s > bankroll:
                        s = bankroll
                else:
                    s = compute_stake(stake, kelly_mult, best["prob"], best["odds"])
                    if s <= 0:
                        continue
                
                p = compute_profit(s, best["odds"], is_win)
                if kelly_mult > 0 and bankroll is not None:
                    bankroll += p
                bet_rows.append({
                    "date": m["date"],
                    "stake": s,
                    "profit": p,
                    "odds": best["odds"],
                    "ev": best["ev"],
                    "edge": best["edge"],
                    "prob": best["prob"],
                    "outcome": best["outcome"]
                })

            # Distribution stats helper
            def get_group_stats(rows, odds_key="odds", outcome_key="outcome"):
                if not rows:
                    return {
                        "count": 0, "avg_odds": 0, "med_odds": 0, "p90_odds": 0,
                        "mix": {"home": 0, "draw": 0, "away": 0}
                    }
                df_g = pd.DataFrame(rows)
                n_g = len(df_g)
                c_g = df_g[outcome_key].value_counts().to_dict()
                return {
                    "count": n_g,
                    "avg_odds": float(round(df_g[odds_key].mean(), 2)),
                    "med_odds": float(round(df_g[odds_key].median(), 2)),
                    "p90_odds": float(round(df_g[odds_key].quantile(0.9), 2)),
                    "mix": {
                        "home": float(round(c_g.get("home", 0) / n_g, 3)),
                        "draw": float(round(c_g.get("draw", 0) / n_g, 3)),
                        "away": float(round(c_g.get("away", 0) / n_g, 3))
                    }
                }

            stats_all_valid = get_group_stats(group_top_prob_all_valid)
            stats_top_passes = get_group_stats(group_top_prob_passes_gate)
            stats_placed = get_group_stats(bet_rows, odds_key="odds", outcome_key="outcome")

            all_valid_definition = "prediction present + labeled outcome + valid odds"

            if not bet_rows:
                cells.append({
                    "min_edge": float(me), "min_ev": float(mv),
                    "bets": 0, "roi": 0.0, "profit": 0.0, "low_sample": True,
                    "n_all_valid": n_total_all_valid,
                    "n_all_valid_matches": n_total_all_valid,
                    "n_any_passes_gate": n_any_passes_gate,
                    "n_top_prob_passes_gate": n_top_prob_passes_gate,
                    "stats_all_valid": stats_all_valid,
                    "stats_top_prob_all_valid": stats_all_valid,
                    "stats_top_passes_gate": stats_top_passes,
                    "stats_placed_bets": stats_placed,
                    "all_valid_definition": all_valid_definition,
                    "pct_h": stats_placed["mix"]["home"],
                    "pct_d": stats_placed["mix"]["draw"],
                    "pct_a": stats_placed["mix"]["away"],
                })
                continue
                
            bdf = pd.DataFrame(bet_rows)
            staked = bdf["stake"].sum()
            profit = bdf["profit"].sum()
            bets = len(bdf)

            cell = {
                "min_edge": float(me),
                "min_ev": float(mv),
                "bets": bets,
                "roi": float(round(profit / staked, 4)) if staked > 0 else 0.0,
                "profit": float(round(profit, 2)),
                "avg_odds": stats_placed["avg_odds"],
                "median_odds": stats_placed["med_odds"],
                "p90_odds": stats_placed["p90_odds"],
                "avg_ev": float(round(bdf["ev"].mean(), 4)),
                "avg_edge": float(round(bdf["edge"].mean(), 4)),
                "low_sample": bool(bets < min_bets),
                
                # NEW Diagnostics
                "n_all_valid": n_total_all_valid,
                "n_all_valid_matches": n_total_all_valid,
                "n_any_passes_gate": n_any_passes_gate,
                "n_top_prob_passes_gate": n_top_prob_passes_gate,
                
                "stats_all_valid": stats_all_valid,
                "stats_top_prob_all_valid": stats_all_valid,
                "stats_top_passes_gate": stats_top_passes,
                "stats_placed_bets": stats_placed,
                "all_valid_definition": all_valid_definition,
                
                # Backward compatibility for existing UI fields
                "pct_h": stats_placed["mix"]["home"],
                "pct_d": stats_placed["mix"]["draw"],
                "pct_a": stats_placed["mix"]["away"],
            }
            
            cell["_raw_bdf"] = bdf 
            cells.append(cell)

            if progress_callback and (
                cell_idx == total_cells or (progress_every > 0 and cell_idx % progress_every == 0)
            ):
                progress_callback({
                    "done": cell_idx,
                    "total": total_cells,
                    "pct": float(round(cell_idx / total_cells, 4)) if total_cells > 0 else 1.0,
                    "min_edge": float(me),
                    "min_ev": float(mv)
                })

    cells.sort(key=lambda x: x["roi"], reverse=True)
    
    ci_computed = 0
    for cell in cells:
        if cell["bets"] >= min_bets and ci_computed < max_ci_cells:
            ci_res = bootstrap_roi(cell["_raw_bdf"], n=bootstrap_n)
            if ci_res["status"] == "success":
                cell["roi_p05"] = float(round(ci_res["roi_p05"], 4))
                cell["roi_p95"] = float(round(ci_res["roi_p95"], 4))
                cell["profit_p05"] = float(round(ci_res["profit_p05"], 2))
                cell["profit_p95"] = float(round(ci_res["profit_p95"], 2))
                ci_computed += 1
        
        if "_raw_bdf" in cell: del cell["_raw_bdf"]
        
    cells.sort(key=lambda x: (x.get("roi_p05", -999), x["roi"]), reverse=True)
    
    result = {
        "cells": cells,
        "summary": {
            "total_matches": len(processed_matches),
            "all_valid_definition": "prediction present + labeled outcome + valid odds",
            "edge_range": edge_range,
            "ev_range": ev_range,
            "min_bets": min_bets,
            "start_date": str(resolved_start.date()),
            "end_date": str(resolved_end.date()),
            "stake": stake,
            "kelly_mult": kelly_mult,
            "selection_mode": selection_mode
        }
    }
    
    save_latest_sweep(result)
    return result
