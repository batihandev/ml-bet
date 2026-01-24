import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from production.predict import load_production_model, predict_ft_1x2
from production.schema import CLASS_MAPPING
from production.backtest import backtest_production_1x2

def verify_mapping():
    print("--- Verifying Mapping ---")
    base_model, calibrator, features = load_production_model()
    
    classes = calibrator.classes_ if calibrator and hasattr(calibrator, "classes_") else base_model.classes_
    print(f"Model classes: {classes}")
    print(f"CLASS_MAPPING values: {set(CLASS_MAPPING.values())}")
    
    if set(classes) != set(CLASS_MAPPING.values()):
        print("ERROR: Model classes do not match CLASS_MAPPING values!")
        sys.exit(1)
    print("SUCCESS: Mapping looks sane.")

def verify_debug_output():
    print("\n--- Testing predict_ft_1x2(debug=1) ---")
    # Load some real data to test
    from dataset.cleaner import load_features
    df = load_features().head(10)
    
    results = predict_ft_1x2(df, debug=1)
    if not results:
        print("ERROR: No results returned from predict_ft_1x2")
        sys.exit(1)
        
    first = results[0]
    if "debug_info" not in first:
        print("ERROR: debug_info missing from first result")
        sys.exit(1)
        
    debug = first["debug_info"]
    print("Debug Info present:")
    print(f"  model_classes: {debug['model_classes']}")
    print(f"  class_mapping: {debug['class_mapping']}")
    print(f"  sample_rows count: {len(debug['sample_rows'])}")
    
    for row in debug['sample_rows']:
        print(f"  Match {row['match_id']}: {row['probs']} -> Top: {row['top_outcome']}")

def verify_backtest_diagnostics():
    print("\n--- Testing backtest_production_1x2 diagnostics ---")
    # Run a small backtest
    res = backtest_production_1x2(start_date="2024-01-01", debug=1)
    summary = res["summary"]
    
    fields = ["n_all_valid", "n_any_passes_gate", "n_top_prob_passes_gate", "stats_all_valid", "stats_top_passes_gate", "stats_placed_bets"]
    missing = [f for f in fields if f not in summary]
    
    if missing:
        print(f"ERROR: Missing diagnostic fields in summary: {missing}")
        sys.exit(1)
        
    print("Diagnostic fields present:")
    print(f"  n_all_valid: {summary['n_all_valid']}")
    print(f"  n_any_passes_gate: {summary['n_any_passes_gate']}")
    print(f"  n_top_prob_passes_gate: {summary['n_top_prob_passes_gate']}")
    print(f"  TopProb Pass Rate: {summary['n_top_prob_passes_gate'] / summary['n_all_valid'] if summary['n_all_valid'] > 0 else 0:.1%}")
    print(f"  Any Pass Rate: {summary['n_any_passes_gate'] / summary['n_all_valid'] if summary['n_all_valid'] > 0 else 0:.1%}")

def verify_sweep_diagnostics():
    print("\n--- Testing run_backtest_sweep diagnostics ---")
    from production.backtest_sweep import run_backtest_sweep
    res = run_backtest_sweep(
        start_date="2024-01-01",
        edge_range=(0.0, 0.05, 0.05),
        ev_range=(0.0, 0.05, 0.05),
        min_bets=10
    )
    cells = res.get("cells", [])
    if not cells:
        print("ERROR: No cells returned from sweep")
        sys.exit(1)
        
    c = cells[0]
    fields = ["n_all_valid", "n_any_passes_gate", "n_top_prob_passes_gate", "stats_all_valid", "stats_top_passes_gate", "stats_placed_bets"]
    missing = [f for f in fields if f not in c]
    if missing:
        print(f"ERROR: Missing diagnostic fields in cell: {missing}")
        sys.exit(1)
        
    print("Sweep Cell Diagnostic fields present:")
    print(f"  n_all_valid: {c['n_all_valid']}")
    print(f"  n_top_prob_passes_gate: {c['n_top_prob_passes_gate']}")

if __name__ == "__main__":
    verify_mapping()
    verify_debug_output()
    verify_backtest_diagnostics()
    verify_sweep_diagnostics()
    print("\nALL VERIFICATIONS PASSED")
