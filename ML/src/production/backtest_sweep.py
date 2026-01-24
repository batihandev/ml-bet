import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from .backtest_utils import (
    build_df_bt, index_predictions, select_best_bet, 
    compute_stake, compute_profit, get_actual_outcome
)
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
    max_ci_cells: int = 50
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
    predictions = predict_ft_1x2(df_bt)
    pred_by_id = index_predictions(predictions)
    
    # Prepare all possible bets info per match
    processed_matches = []
    for _, match_row in df_bt.iterrows():
        mid = int(match_row["match_id"])
        if mid not in pred_by_id: continue
        actual = get_actual_outcome(match_row)
        if actual is None: continue
        processed_matches.append({
            "id": mid,
            "date": match_row["match_date"],
            "actual": actual,
            "metrics": pred_by_id[mid]["metrics"]
        })
    
    # Generate grid
    edges = np.round(np.arange(edge_range[0], edge_range[1] + 0.0001, edge_range[2]), 3)
    evs = np.round(np.arange(ev_range[0], ev_range[1] + 0.0001, ev_range[2]), 3)
    
    cells = []
    
    for me in edges:
        for mv in evs:
            bet_rows = []
            for m in processed_matches:
                best = select_best_bet(m["metrics"], me, mv)
                if not best: continue
                
                from .schema import CLASS_MAPPING
                is_win = (m["actual"] == CLASS_MAPPING[best["outcome"]])
                
                s = compute_stake(stake, kelly_mult, best["prob"], best["odds"])
                if s <= 0: continue
                
                p = compute_profit(s, best["odds"], is_win)
                bet_rows.append({
                    "date": m["date"],
                    "stake": s,
                    "profit": p,
                    "odds": best["odds"],
                    "ev": best["ev"],
                    "edge": best["edge"]
                })
                
            if not bet_rows:
                cells.append({
                    "min_edge": float(me), "min_ev": float(mv),
                    "bets": 0, "roi": 0.0, "profit": 0.0, "low_sample": True
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
                "avg_odds": float(round(bdf["odds"].mean(), 2)),
                "avg_ev": float(round(bdf["ev"].mean(), 4)),
                "low_sample": bool(bets < min_bets)
            }
            
            cell["_raw_bdf"] = bdf 
            cells.append(cell)

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
            "edge_range": edge_range,
            "ev_range": ev_range,
            "min_bets": min_bets,
            "start_date": str(resolved_start.date()),
            "end_date": str(resolved_end.date()),
            "stake": stake,
            "kelly_mult": kelly_mult
        }
    }
    
    save_latest_sweep(result)
    return result
