import pandas as pd
from production.backtest import backtest_production_1x2
from production.backtest_sweep import run_backtest_sweep

def verify_consistency():
    start_date = "2025-01-01"
    end_date = "2025-06-01"
    min_edge = 0.01
    min_ev = 0.0
    selection_mode = "top_prob_only"
    
    print(f"Running backtest with selection_mode={selection_mode}...")
    bt_res = backtest_production_1x2(
        start_date=start_date,
        end_date=end_date,
        min_edge=min_edge,
        min_ev=min_ev,
        selection_mode=selection_mode
    )
    bt_bets = bt_res["summary"]["total_bets"]
    bt_profit = bt_res["summary"]["total_profit"]
    
    print(f"Running sweep with selection_mode={selection_mode}...")
    sweep_res = run_backtest_sweep(
        start_date=start_date,
        end_date=end_date,
        edge_range=(min_edge, min_edge, 0.01),
        ev_range=(min_ev, min_ev, 0.01),
        selection_mode=selection_mode
    )
    
    # Sweep results are in cells (normalized by grid generation)
    # The grid generator in backtest_sweep uses np.round(..., 3)
    target_edge = float(round(min_edge, 3))
    target_ev = float(round(min_ev, 3))
    
    cell = next((c for c in sweep_res["cells"] if abs(c["min_edge"] - target_edge) < 1e-6 and abs(c["min_ev"] - target_ev) < 1e-6), None)
    if not cell:
        print(f"Cell NOT FOUND for edge={target_edge}, ev={target_ev}")
        print(f"Available cells: {[(c['min_edge'], c['min_ev']) for c in sweep_res['cells'][:5]]}")
        return

    sweep_bets = cell["bets"]
    sweep_profit = cell["profit"]
    
    print(f"Backtest: {bt_bets} bets, {bt_profit} profit")
    print(f"Sweep Cell: {sweep_bets} bets, {sweep_profit} profit")
    
    if bt_bets != sweep_bets or abs(bt_profit - sweep_profit) > 0.01:
        print("BT Summary:", bt_res["summary"])
        print("Sweep Cell:", cell)
    
    assert bt_bets == sweep_bets, f"Bet count mismatch: {bt_bets} vs {sweep_bets}"
    assert abs(bt_profit - sweep_profit) < 0.01, f"Profit mismatch: {bt_profit} vs {sweep_profit}"
    print("Consistency check PASSED!")

    # Test best_ev mode too
    selection_mode = "best_ev"
    print(f"\nRunning backtest with selection_mode={selection_mode}...")
    bt_res = backtest_production_1x2(
        start_date=start_date,
        end_date=end_date,
        min_edge=min_edge,
        min_ev=min_ev,
        selection_mode=selection_mode
    )
    bt_bets = bt_res["summary"]["total_bets"]
    bt_profit = bt_res["summary"]["total_profit"]
    
    print(f"Running sweep with selection_mode={selection_mode}...")
    sweep_res = run_backtest_sweep(
        start_date=start_date,
        end_date=end_date,
        edge_range=(min_edge, min_edge, 0.01),
        ev_range=(min_ev, min_ev, 0.01),
        selection_mode=selection_mode
    )
    
    cell = next(c for c in sweep_res["cells"] if c["min_edge"] == min_edge and c["min_ev"] == min_ev)
    sweep_bets = cell["bets"]
    sweep_profit = cell["profit"]
    
    print(f"Backtest: {bt_bets} bets, {bt_profit} profit")
    print(f"Sweep Cell: {sweep_bets} bets, {sweep_profit} profit")
    
    assert bt_bets == sweep_bets, f"Bet count mismatch: {bt_bets} vs {sweep_bets}"
    assert abs(bt_profit - sweep_profit) < 0.01, f"Profit mismatch: {bt_profit} vs {sweep_profit}"
    print("Consistency check PASSED (best_ev)!")

if __name__ == "__main__":
    verify_consistency()
