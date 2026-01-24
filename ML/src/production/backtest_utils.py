import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from .schema import CLASS_MAPPING
from .utils_market import is_valid_odds

def build_df_bt(df_all: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """
    Filter and prepare the backtest dataframe.
    """
    df_window = df_all[
        (df_all["match_date"] >= start_date) & 
        (df_all["match_date"] <= end_date)
    ].copy()

    if df_window.empty:
        return pd.DataFrame()

    # Filter to labeled rows
    df_bt = df_window[
        df_window["ft_home_goals"].notna() & 
        df_window["ft_away_goals"].notna()
    ].copy()
    
    if df_bt.empty:
        return pd.DataFrame()

    # Normalize match_id
    df_bt["match_id"] = df_bt["match_id"].astype("int64")
    
    # Sort deterministically
    df_bt = df_bt.sort_values(["match_date", "match_id"])
    
    if df_bt["match_id"].duplicated().any():
        raise ValueError("Duplicate match_id in backtest window")
        
    return df_bt

def index_predictions(predictions: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """
    Create a robust lookup map for predictions by match ID.
    """
    pred_by_id = {}
    for r in predictions:
        mid = int(r["match_id"])
        if mid in pred_by_id:
            raise ValueError(f"Duplicate prediction for match_id={mid}")
        pred_by_id[mid] = r
    return pred_by_id

def select_best_bet(metrics: List[Dict[str, Any]], min_edge: float, min_ev: float) -> Optional[Dict[str, Any]]:
    """
    Apply gating and pick the best bet (highest EV).
    EV Definition (Canonical): EV = P_model * Odds - 1
    Edge Definition: Edge = P_model - P_implied_normalized
    """
    valid_bets = [
        m for m in metrics 
        if m["edge"] >= min_edge and m["ev"] >= min_ev and m["odds"] > 1.0
    ]
    
    if not valid_bets:
        return None
        
    return max(valid_bets, key=lambda x: x["ev"])

def compute_stake(stake_base: float, kelly_mult: float, prob: float, odds: float) -> float:
    """
    Compute stake amount based on flat or fractional Kelly.
    """
    if kelly_mult > 0:
        # Kelly % = (p * b - q) / b where b = odds - 1
        # Simplified: (p * odds - 1) / (odds - 1)
        kelly_f = (prob * odds - 1.0) / (odds - 1.0)
        if kelly_f <= 0:
            return 0.0
        return stake_base * kelly_mult * kelly_f
    return stake_base

def compute_profit(stake: float, odds: float, is_win: bool) -> float:
    """
    Compute net profit/loss for a bet.
    """
    if is_win:
        return stake * (odds - 1.0)
    return -stake

def get_actual_outcome(match_row: pd.Series) -> Optional[int]:
    """
    Convert goals to CLASS_MAPPING index.
    """
    h = match_row.get("ft_home_goals")
    a = match_row.get("ft_away_goals")
    if pd.isna(h) or pd.isna(a):
        return None
    if h > a: return CLASS_MAPPING["home"]
    if h < a: return CLASS_MAPPING["away"]
    return CLASS_MAPPING["draw"]

def compute_max_drawdown(equity_series: pd.Series, mode: str = "flat") -> float:
    """
    Compute max drawdown from an equity curve.
    - flat: equity_series is cumulative profit (additive).
    - kelly: equity_series is bankroll (multiplicative).
    """
    if equity_series.empty:
        return 0.0
        
    if mode == "flat":
        # For flat staking, MDD is the largest drop from a peak in units
        running_max = equity_series.cummax()
        drawdown = running_max - equity_series
        return float(drawdown.max())
    else:
        # For multiplicative (bankroll), MDD is (Peak - Trough) / Peak
        running_max = equity_series.cummax()
        drawdown = (running_max - equity_series) / running_max
        return float(drawdown.max())
