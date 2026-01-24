import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add ML/src to path
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR / "ML" / "src"))

from production.schema import TARGET_COL, CLASS_MAPPING, LABEL_MAPPING
from production.train import prepare_1x2_target 
from dataset.cleaner import clean_training_data, select_feature_columns

PROCESSED_DIR = ROOT_DIR / "ML" / "data" / "processed"

def verify_draws():
    print("=== 1. Verify Label Correctness ===")
    
    # Load data
    feat_path = PROCESSED_DIR / "features.csv"
    if not feat_path.exists():
        print(f"ERROR: Feature file not found: {feat_path}")
        return

    df = pd.read_csv(feat_path, parse_dates=["match_date"], low_memory=False)
    print(f"Loaded {len(df)} rows.")

    # Apply target preparation logic explicitly
    df = prepare_1x2_target(df)
    
    # Check for NaNs in target
    n_na = df[TARGET_COL].isna().sum()
    print(f"Rows with NaN target: {n_na} ({(n_na/len(df))*100:.2f}%)")
    
    # Drop NaNs for stats
    df_clean = df.dropna(subset=[TARGET_COL]).copy()
    
    # Counts Overall
    counts = df_clean[TARGET_COL].value_counts().sort_index()
    print("\nTarget Counts (Overall):")
    for cls_idx, count in counts.items():
        label = LABEL_MAPPING.get(int(cls_idx), "UNKNOWN")
        pct = (count / len(df_clean)) * 100
        print(f"  Class {cls_idx} ({label}): {count} ({pct:.2f}%)")
        
    # Per Season
    df_clean["season_year"] = df_clean["match_date"].dt.year
    print("\nTarget Distribution by Year (Last 3 years):")
    for year in sorted(df_clean["season_year"].unique())[-3:]:
        year_df = df_clean[df_clean["season_year"] == year]
        c = year_df[TARGET_COL].value_counts().sort_index()
        total = len(year_df)
        print(f"  {year}: Total {total}")
        for cls_idx, count in c.items():
            label = LABEL_MAPPING.get(int(cls_idx), "UNKNOWN")
            print(f"    {label}: {count} ({count/total:.2%})")

    # Spot-check 50 random "draw" rows
    print("\nSpot-check 50 random DRAW rows:")
    draw_val = CLASS_MAPPING["draw"]
    draws = df_clean[df_clean[TARGET_COL] == draw_val]
    
    if len(draws) > 0:
        sample = draws.sample(min(50, len(draws)), random_state=42)
        errors = 0
        for idx, row in sample.iterrows():
            h = row.get("ft_home_goals")
            a = row.get("ft_away_goals")
            if h != a:
                print(f"  ERROR: Row {idx} labeled DRAW but {h} != {a}")
                errors += 1
        if errors == 0:
            print("  SUCCESS: All 50 random draw rows have equal home/away goals.")
        else:
            print(f"  FAILURE: Found {errors} mislabeled rows.")
    else:
        print("  WARNING: No draws found to check.")

    print("\n=== 2. Confirm Class Index Mapping ===")
    print(f"Current CLASS_MAPPING: {CLASS_MAPPING}")
    print(f"Current LABEL_MAPPING: {LABEL_MAPPING}")
    
    # Check consistency
    print("Checking consistency on full dataset...")
    # Home check
    n_home = len(df_clean[(df_clean["ft_home_goals"] > df_clean["ft_away_goals"])])
    n_home_lbl = len(df_clean[df_clean[TARGET_COL] == CLASS_MAPPING["home"]])
    
    n_away = len(df_clean[(df_clean["ft_home_goals"] < df_clean["ft_away_goals"])])
    n_away_lbl = len(df_clean[TARGET_COL] == CLASS_MAPPING["away"]) # Wait, this logic is flawed if I don't filter.
    
    # Correct consistency check:
    # Filter where Target matches Home Class, check if goals align
    home_mask = df_clean[TARGET_COL] == CLASS_MAPPING["home"]
    home_failures = df_clean[home_mask & ~(df_clean["ft_home_goals"] > df_clean["ft_away_goals"])]
    
    draw_mask = df_clean[TARGET_COL] == CLASS_MAPPING["draw"]
    draw_failures = df_clean[draw_mask & ~(df_clean["ft_home_goals"] == df_clean["ft_away_goals"])]
    
    away_mask = df_clean[TARGET_COL] == CLASS_MAPPING["away"]
    away_failures = df_clean[away_mask & ~(df_clean["ft_home_goals"] < df_clean["ft_away_goals"])]
    
    print(f"  Home Mismatches: {len(home_failures)}")
    print(f"  Draw Mismatches: {len(draw_failures)}")
    print(f"  Away Mismatches: {len(away_failures)}")
    
    if len(home_failures) + len(draw_failures) + len(away_failures) == 0:
        print("  SUCCESS: Class mapping is perfectly consistent with goals.")
    else:
        print("  FAILURE: Class mapping has inconsistencies.")

    print("\n=== 4. Check Odds Features Dominance (Baseline) ===")
    
    valid_odds = df_clean.dropna(subset=["odd_home", "odd_draw", "odd_away"]).copy()
    print(f"Rows with valid odds: {len(valid_odds)}")
    
    def get_bookmaker_choice(row):
        odds = [row["odd_home"], row["odd_draw"], row["odd_away"]]
        min_odd = min(odds)
        if min_odd == row["odd_home"]: return CLASS_MAPPING["home"]
        if min_odd == row["odd_draw"]: return CLASS_MAPPING["draw"]
        return CLASS_MAPPING["away"]

    valid_odds["bookmaker_pred"] = valid_odds.apply(get_bookmaker_choice, axis=1)
    acc = (valid_odds["bookmaker_pred"] == valid_odds[TARGET_COL]).mean()
    
    print(f"Baseline Bookmaker Favourite Accuracy: {acc:.4f}")
    
    # Log loss of bookmaker implied probs
    from sklearn.metrics import log_loss
    
    # Filter out rows with invalid odds (<= 1)
    valid_odds = valid_odds[
        (valid_odds["odd_home"] > 1) & 
        (valid_odds["odd_draw"] > 1) & 
        (valid_odds["odd_away"] > 1)
    ].copy()
    print(f"Rows with valid odds (>1.0): {len(valid_odds)}")
    
    def get_p_imp(row):
        raw = [1/row["odd_home"], 1/row["odd_draw"], 1/row["odd_away"]]
        s = sum(raw)
        return [r/s for r in raw]
        
    p_imp = valid_odds.apply(get_p_imp, axis=1).tolist()
    y_true = valid_odds[TARGET_COL].values
    
    # Align probs to classes 0, 1, 2. Bookmaker odds are usually H, D, A.
    # Our mapping: Home=0, Draw=1, Away=2.
    # So if odds are [H, D, A], index 0 is Home, 1 is Draw, 2 is Away. Matches our mapping.
    
    baseline_loss = log_loss(y_true, p_imp)
    print(f"Baseline Bookmaker Log Loss: {baseline_loss:.4f}")

if __name__ == "__main__":
    verify_draws()
