import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add ML/src to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from production.backtest import backtest_production_1x2
from production.backtest_utils import (
    build_df_bt, index_predictions, select_best_bet, 
    compute_stake, compute_profit, get_actual_outcome
)
from production.predict import predict_ft_1x2
from dataset.cleaner import load_features

def run_identity_sanity():
    print("--- Running Identity Sanity: P_model = P_implied_normalized ---")
    df_all = load_features()
    # Use a recent window
    end = df_all["match_date"].max()
    start = end - pd.Timedelta(days=90)
    
    df_bt = build_df_bt(df_all, start, end)
    if df_bt.empty:
        print("Error: No data for sanity check")
        return False
        
    predictions = predict_ft_1x2(df_bt)
    
    # Override predictions to match normalized implied probabilities
    for p in predictions:
        for m in p["metrics"]:
            # Set model prob to implied normalized prob
            m["prob"] = m["pimp"]
            # Recalculate edge and ev
            m["edge"] = 0.0 # p - pimp
            # ev = p * odds - 1. If p is normalized pimp, ev should be (1/margin) - 1
            m["ev"] = m["prob"] * m["odds"] - 1.0
            
    pred_by_id = index_predictions(predictions)
    
    # Test 1: min_edge > 0 should place 0 bets
    valid_bets_edge = []
    for _, row in df_bt.iterrows():
        best = select_best_bet(pred_by_id[int(row["match_id"])]["metrics"], min_edge=0.001, min_ev=-1.0)
        if best: valid_bets_edge.append(best)
        
    print(f"Bets with min_edge=0.001: {len(valid_bets_edge)} (Expect 0)")
    if len(valid_bets_edge) > 0:
        print("FAIL: Identity sanity (edge)")
        return False

    # Test 2: min_ev > 0 should place 0 bets (since 1/margin - 1 is negative)
    valid_bets_ev = []
    for _, row in df_bt.iterrows():
        best = select_best_bet(pred_by_id[int(row["match_id"])]["metrics"], min_edge=-1.0, min_ev=0.0)
        if best: valid_bets_ev.append(best)
        
    print(f"Bets with min_ev=0.0: {len(valid_bets_ev)} (Expect 0)")
    if len(valid_bets_ev) > 0:
        print("FAIL: Identity sanity (ev)")
        return False

    # Test 3: blind betting (no gating) should have negative ROI
    bet_rows = []
    for _, row in df_bt.iterrows():
        best = select_best_bet(pred_by_id[int(row["match_id"])]["metrics"], min_edge=-1.0, min_ev=-1.0)
        if best:
            actual = get_actual_outcome(row)
            if actual is not None:
                is_win = (actual == best["outcome"]) # This is just a simulation
                # In identity sanity, we don't expect profit, we expect ROI ~ 1/margin - 1
                # The actual outcomes don't matter for the *expected* ROI, 
                # but we'll run it to see the distribution.
                from production.schema import CLASS_MAPPING
                is_win = (actual == CLASS_MAPPING[best["outcome"]])
                profit = compute_profit(1.0, best["odds"], is_win)
                bet_rows.append({"stake": 1.0, "profit": profit})
                
    if bet_rows:
        rdf = pd.DataFrame(bet_rows)
        roi = rdf["profit"].sum() / rdf["stake"].sum()
        print(f"Blind ROI with P_model=P_imp: {roi:.4f} (Expect negative ~ margin)")
        if roi > 0.05: # High threshold for noise since it's a small sample
            print("FAIL: Identity sanity (ROI too high)")
            return False
            
    print("SUCCESS: Identity sanity passed")
    return True

def run_random_sanity():
    print("--- Running Random Model Sanity ---")
    df_all = load_features()
    end = df_all["match_date"].max()
    start = end - pd.Timedelta(days=90)
    df_bt = build_df_bt(df_all, start, end)
    
    predictions = predict_ft_1x2(df_bt)
    
    # Override with random probabilities
    for p in predictions:
        r = np.random.dirichlet([1, 1, 1])
        # home, draw, away usually in this order
        p_map = {"home": r[0], "draw": r[1], "away": r[2]}
        for m in p["metrics"]:
            m["prob"] = p_map[m["outcome"]]
            m["edge"] = m["prob"] - m["pimp"]
            m["ev"] = m["prob"] * m["odds"] - 1.0
            
    pred_by_id = index_predictions(predictions)
    
    bet_rows = []
    for _, row in df_bt.iterrows():
        best = select_best_bet(pred_by_id[int(row["match_id"])]["metrics"], min_edge=0.02, min_ev=0.0)
        if best:
            actual = get_actual_outcome(row)
            if actual is not None:
                from production.schema import CLASS_MAPPING
                is_win = (actual == CLASS_MAPPING[best["outcome"]])
                profit = compute_profit(1.0, best["odds"], is_win)
                bet_rows.append({"stake": 1.0, "profit": profit})
                
    if bet_rows:
        rdf = pd.DataFrame(bet_rows)
        roi = rdf["profit"].sum() / rdf["stake"].sum()
        print(f"Random ROI with gating: {roi:.4f} (Expect negative)")
        if roi > 0.15: # Random noise can be high
             print("WARNING: Random ROI surprisingly high, but could be luck. Re-running with more samples recommended.")
    else:
        print("No bets placed with random model (OK)")

    print("SUCCESS: Random sanity complete")
    return True

if __name__ == "__main__":
    s1 = run_identity_sanity()
    s2 = run_random_sanity()
    if not (s1 and s2):
        sys.exit(1)
