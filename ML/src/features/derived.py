import pandas as pd
import numpy as np
from typing import List

def add_draw_closeness_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features for draw predictability:
    - Gap features (home - away)
    - Absolute gap features (abs(home - away))
    
    Dynamically scans for paired columns (home_X, away_X) matching specific patterns.
    """
    df = df.copy()
    
    # Identify candidate pairs dynamically
    # We look for columns starting with "home_" and check if corresponding "away_" exists
    candidates = [c for c in df.columns if c.startswith("home_")]
    
    for home_col in candidates:
        suffix = home_col[len("home_"):]
        away_col = f"away_{suffix}"
        
        if away_col not in df.columns:
            continue
            
        # Whitelist check
        # 1. Special case: attack strength
        # 2. Rolling form features with specific metrics
        is_whitelisted = False
        
        if suffix == "attack_strength":
            is_whitelisted = True
        elif home_col.startswith("home_form"):
            # check for specific metrics in the suffix
            allowed_metrics = ["gd", "ppg", "goals", "win_rate", "draw_rate", "goals_for", "goals_against"]
            if any(m in suffix for m in allowed_metrics):
                is_whitelisted = True
        
        if not is_whitelisted:
            continue
            
        # Create gap names
        gap_col = f"gap_{suffix}"
        abs_gap_col = f"abs_gap_{suffix}"
        
        # Check duplication
        if gap_col in df.columns:
            continue

        # Safe PPG calculation if needed (though user plan mentioned computing it from rates, 
        # the whitelist implies we match existing columns. If "ppg" is in the name, we assume it's valid 
        # OR we might need to derive PPG if it doesn't exist but the rates do. 
        # The user instruction was: "Compute valid PPG only from safe rolling columns".
        # If the column `home_formX_ppg` doesn't exist, we can't 'pair' it.
        # So we assume the user might have meant: IF we find home_formX_win_rate etc, WE create gap_ppg?
        # Let's re-read carefully: "Implement derived gaps by pairing existing columns... specifically targeting... form*_ppg"
        # If 'home_form*_ppg' isn't in df, we won't find it here. 
        # BUT the plan says "Compute valid PPG only from safe rolling columns". 
        # This implies we might need to CREATE the ppg columns first? 
        # Or does it mean "If we are creating a gap for ppg, make sure the ppg source was safe"?
        # Given "whitelist pairings... form*_ppg", I'll assume if `home_formX_ppg` exists, we use it.
        # IF IT DOES NOT, we might need to create it. 
        # However, to be safe and strictly follow "pair existing columns", I will only gap what exists.
        # Wait, the user said "Implement derived features... specifically abs(home_form*_ppg - away_form*_ppg)".
        # If those don't exist, I should probably CREATE them on the fly?
        # A clearer approach for PPG:
        # If we see `home_formX_ft_win_rate` and `home_formX_ft_draw_rate`, we can DERIVE PPG gaps.
        
        # Let's stick to the simpler "Gap what exists" first, but add a special block for PPG derivation if the raw rates exist.
        
        # 1. Standard Gap Calculation for existing pair
        val_home = pd.to_numeric(df[home_col], errors="coerce")
        val_away = pd.to_numeric(df[away_col], errors="coerce")
        
        df[gap_col] = val_home - val_away
        df[abs_gap_col] = df[gap_col].abs()

    # Special Pass: PPG Derivation from Rates if they exist but PPG itself doesn't
    # Pattern: form{w}_ft_win_rate / form{w}_ft_draw_rate
    # We want to create gap_form{w}_ppg
    
    # Scan for win rates
    win_cols = [c for c in df.columns if "form" in c and "_ft_win_rate" in c and c.startswith("home_")]
    for home_win_col in win_cols:
        # distinct window extraction
        # format: home_form{w}_ft_win_rate -> window is between "form" and "_"??
        # safer: just replace strings
        base_suffix = home_win_col.replace("home_", "").replace("_ft_win_rate", "") 
        # base_suffix likely like "form5"
        
        home_draw_col = home_win_col.replace("ft_win_rate", "ft_draw_rate")
        
        away_win_col = home_win_col.replace("home_", "away_")
        away_draw_col = home_draw_col.replace("home_", "away_")
        
        if all(c in df.columns for c in [home_draw_col, away_win_col, away_draw_col]):
             ppg_gap_name = f"gap_{base_suffix}_ppg"
             ppg_abs_gap_name = f"abs_gap_{base_suffix}_ppg"
             
             if ppg_gap_name in df.columns:
                 continue
                 
             h_win = pd.to_numeric(df[home_win_col], errors="coerce")
             h_draw = pd.to_numeric(df[home_draw_col], errors="coerce")
             a_win = pd.to_numeric(df[away_win_col], errors="coerce")
             a_draw = pd.to_numeric(df[away_draw_col], errors="coerce")
             
             home_ppg = 3 * h_win + 1 * h_draw
             away_ppg = 3 * a_win + 1 * a_draw
             
             df[ppg_gap_name] = home_ppg - away_ppg
             df[ppg_abs_gap_name] = df[ppg_gap_name].abs()

    return df
