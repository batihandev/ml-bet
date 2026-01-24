from production.backtest import backtest_production_1x2
import pandas as pd

def test_selection_mode_difference():
    # Use a range where we expect multiple outcomes might pass the thresholds
    # or where the top prob might fail but others pass.
    start_date = "2025-01-01"
    end_date = "2025-06-01"
    min_edge = 0.01
    min_ev = 0.0
    
    print(f"Testing differences for range {start_date} to {end_date}...")
    
    res_ev = backtest_production_1x2(
        start_date=start_date,
        end_date=end_date,
        min_edge=min_edge,
        min_ev=min_ev,
        selection_mode="best_ev"
    )
    
    res_top = backtest_production_1x2(
        start_date=start_date,
        end_date=end_date,
        min_edge=min_edge,
        min_ev=min_ev,
        selection_mode="top_prob_only"
    )
    
    bets_ev = res_ev["summary"]["total_bets"]
    bets_top = res_top["summary"]["total_bets"]
    
    profit_ev = res_ev["summary"]["total_profit"]
    profit_top = res_top["summary"]["total_profit"]
    
    print(f"Best EV:  {bets_ev} bets, Profit: {profit_ev}")
    print(f"Top Prob: {bets_top} bets, Profit: {profit_top}")
    
    if bets_ev != bets_top:
        print("SUCCESS: Selection modes produced different bet counts.")
    else:
        print("WARNING: Selection modes produced identical bet counts. This might be normal if the top-prob bet is always the best EV bet, or if other outcomes never pass thresholds.")

    # Let's check a few specific matches if possible
    if not res_ev["bets_df"].empty and not res_top["bets_df"].empty:
        df_ev = res_ev["bets_df"]
        df_top = res_top["bets_df"]
        
        # Matches in EV but not in Top
        ev_ids = set(df_ev["match_id"])
        top_ids = set(df_top["match_id"])
        
        diff = ev_ids - top_ids
        print(f"Matches in Best EV but NOT in Top Prob: {len(diff)}")
        
        if diff:
            sample_id = list(diff)[0]
            bet_ev = df_ev[df_ev["match_id"] == sample_id].iloc[0]
            print(f"Sample mismatch (Match ID {sample_id}):")
            print(f"  Best EV chose: {bet_ev['selected_outcome']} (EV: {bet_ev['selected_ev']}, Prob: {bet_ev['selected_prob']})")
            print(f"  Top Prob chose: NOTHING (likely top prob failed thresholds)")

if __name__ == "__main__":
    test_selection_mode_difference()
