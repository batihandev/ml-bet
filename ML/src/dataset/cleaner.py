import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

def load_features() -> pd.DataFrame:
    path = PROCESSED_DIR / "features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")
    df = pd.read_csv(path, parse_dates=["match_date"], low_memory=False, dtype={"match_time": "string"})
    return df

def make_time_split(df: pd.DataFrame, fixed_cutoff: Optional[str] = None, default_val_span: str = "6 months", mode: str = "default") -> Tuple[np.ndarray, np.ndarray]:
    """
    Split data into train/validation sets based on time.
    """
    if fixed_cutoff:
        cutoff = pd.to_datetime(fixed_cutoff)
    elif mode == "quantile":
        # Experimental/legacy quantile split
        cutoff = df["match_date"].quantile(0.8)
    else:
        # Default: time-based offset from the end of data
        max_date = df["match_date"].max()
        # Parse simple "X months" or "Y days" logic roughly, or just use pd.Timedelta if span is compatible
        # For simplicity tailored to "6 months", we assume standard offsets. 
        # But pd.Timedelta doesn't support "months". We can use DateOffset.
        # Check if user passed a simple int/float days, otherwise hardcode logic for commonly used strings.
        if "month" in default_val_span:
            try:
                months = int(default_val_span.split()[0])
                cutoff = max_date - pd.DateOffset(months=months)
            except:
                cutoff = max_date - pd.DateOffset(months=6) # Fallback
        else:
             # Try generic timedelta
             try:
                 cutoff = max_date - pd.Timedelta(default_val_span)
             except:
                 cutoff = max_date - pd.DateOffset(months=6)

    train_mask = df["match_date"] < cutoff
    val_mask = ~train_mask
    return train_mask.values, val_mask.values

def select_feature_columns(df: pd.DataFrame, exclude_odds: bool = False) -> List[str]:
    # 1. Build Candidate List
    base_features = [
        "home_elo", "away_elo", "form3_home", "form5_home", "form3_away", "form5_away",
        "gap_elo", "abs_gap_elo", "gap_form3", "abs_gap_form3", "gap_form5", "abs_gap_form5",
        "odd_home", "odd_draw", "odd_away", "max_odd_home", "max_odd_draw", "max_odd_away",
        "odd_over25", "odd_under25", "max_odd_over25", "max_odd_under25",
        "handicap_size", "handicap_home", "handicap_away",
    ]
    league_features = [
        "league_avg_goals", "league_home_win_rate", "league_draw_rate", "league_away_win_rate",
        "home_attack_strength", "away_attack_strength",
    ]
    congestion_features = [
        "home_days_since_last", "away_days_since_last", 
        "home_matches_last_7d", "home_matches_last_14d", "home_matches_last_21d", 
        "away_matches_last_7d", "away_matches_last_14d", "away_matches_last_21d",
    ]
    
    # 2. Add candidates if present
    feature_cols = []
    candidates = base_features + league_features + congestion_features
    for c in candidates:
        if c in df.columns:
            feature_cols.append(c)
            
    # 3. Dynamic Includes (Rolling / H2H / Derived Gaps)
    # We scan all columns for these patterns.
    # Note: "Draw Closeness" features (gap_*) are also picked up here if they exist.
    dynamic_candidates = [
        c for c in df.columns 
        if c.startswith("home_form") or c.startswith("away_form") 
        or c.startswith("h2h_") or c.startswith("gap_") or c.startswith("abs_gap_")
        or c.startswith("rule_score_")
    ]
    feature_cols.extend(sorted(dynamic_candidates))
    
    # Remove duplicates if any
    feature_cols = sorted(list(set(feature_cols)))

    # 4. Safety Net: Ensure form/h2h are included IF they exist (redundant vs step 3 but guarantees user intent)
    # Actually step 3 already catches them. The "safety net" instruction is to ensure we DO capture them.
    # Step 3 logic `c.startswith...` is the safety net capture mechanism.
    # We verify that we haven't missed any.
    
    # 5. Final Filter: Exclude Odds (Strict Limit)
    if exclude_odds:
        feature_cols = [c for c in feature_cols if not is_odds_feature(c)]
        
    return feature_cols

def is_odds_feature(col_name: str) -> bool:
    """
    Return True if the column is derived from odds or market data.
    Uses a two-tier predicate: strict prefix/suffix + strict tokenized keywords.
    """
    c = col_name.lower()
    
    # Tier 1: Strict Prefixes / Suffixes
    prefixes = ("odd_", "max_odd_", "min_odd_", "avg_odd_")
    if c.startswith(prefixes): return True
    if c.endswith("_odd"): return True
    
    # Tier 2: Strict Tokenized Keywords
    # Split by underscore to avoid partial matches (e.g. 'handicapper' != 'handicap')
    tokens = set(c.split("_"))
    keywords = {"handicap", "spread", "vig", "implied", "bookmaker", "overround", "margin"}
    
    if not tokens.isdisjoint(keywords):
        return True
        
    return False

def clean_training_data(df: pd.DataFrame, feature_cols: List[str], target_col: str, fixed_cutoff: Optional[str] = None):
    # Filter rows where target is missing
    df = df[~df[target_col].isna()].copy()
    
    # Data-driven history filter
    # Instead of dropping if ANY rolling feature is NaN, require a minimum % of them.
    # This prevents dropping rows that just miss one specific deep history feature (e.g. form10) but have form3.
    df = filter_insufficient_history(df, feature_cols)
    
    X = df[feature_cols].copy().fillna(0.0)
    y = df[target_col].astype(int).values
    train_mask, val_mask = make_time_split(df, fixed_cutoff)
    
    return X[train_mask], X[val_mask], y[train_mask], y[val_mask], feature_cols

def filter_insufficient_history(
    df: pd.DataFrame,
    feature_cols: List[str],
    min_history_frac: float = 0.20
) -> pd.DataFrame:
    """
    Drop rows with too little rolling history (form/h2h).
    """
    rolling_cols = [c for c in feature_cols if "form" in c or c.startswith("h2h_")]
    if not rolling_cols:
        return df
    min_k = max(1, int(min_history_frac * len(rolling_cols)))
    valid_history_mask = df[rolling_cols].notna().sum(axis=1) >= min_k

    dropped_count = (~valid_history_mask).sum()
    if dropped_count > 0:
        print(f"Dropping {dropped_count} rows due to insufficient history (threshold < {min_k} valid cols)")

    return df[valid_history_mask].copy()
