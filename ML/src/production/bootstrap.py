import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple

def bootstrap_roi(
    bets_df: pd.DataFrame, 
    n: int = 1000, 
    seed: int = 42,
    mode: str = "date_block",
    min_bets: int = 50
) -> Dict[str, Any]:
    """
    Perform bootstrap resampling to compute confidence intervals for ROI and Profit.
    Uses 'date_block' bootstrap by default to account for intra-day correlations.
    """
    n_bets = len(bets_df)
    if n_bets < min_bets:
        return {"status": "insufficient sample", "n_bets": n_bets}

    np.random.seed(seed)
    
    roi_samples = []
    profit_samples = []
    
    if mode == "date_block":
        # Group bets by date to preserve intra-day correlations
        dates = bets_df["date"].unique()
        n_dates = len(dates)
        
        # Pre-grouping for performance
        date_groups = {date: group for date, group in bets_df.groupby("date")}
        
        for _ in range(n):
            # Resample dates with replacement
            resampled_dates = np.random.choice(dates, size=n_dates, replace=True)
            
            # Combine all bets for these dates
            # Note: We use list comprehension and pd.concat for efficiency
            sample_bets = pd.concat([date_groups[d] for d in resampled_dates], ignore_index=True)
            
            staked = sample_bets["stake"].sum()
            profit = sample_bets["profit"].sum()
            
            roi_samples.append(profit / staked if staked > 0 else 0)
            profit_samples.append(profit)
    else:
        # standard i.i.d. bootstrap
        for _ in range(n):
            sample_indices = np.random.choice(bets_df.index, size=n_bets, replace=True)
            sample_bets = bets_df.loc[sample_indices]
            
            staked = sample_bets["stake"].sum()
            profit = sample_bets["profit"].sum()
            
            roi_samples.append(profit / staked if staked > 0 else 0)
            profit_samples.append(profit)

    roi_samples = np.array(roi_samples)
    profit_samples = np.array(profit_samples)

    return {
        "status": "success",
        "n_bets": n_bets,
        "roi_mean": float(np.mean(roi_samples)),
        "roi_p05": float(np.percentile(roi_samples, 5)),
        "roi_p50": float(np.percentile(roi_samples, 50)),
        "roi_p95": float(np.percentile(roi_samples, 95)),
        "profit_mean": float(np.mean(profit_samples)),
        "profit_p05": float(np.percentile(profit_samples, 5)),
        "profit_p50": float(np.percentile(profit_samples, 50)),
        "profit_p95": float(np.percentile(profit_samples, 95)),
    }
