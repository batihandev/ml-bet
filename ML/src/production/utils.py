import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import List, Tuple, Dict, Any

def build_time_folds(df, train_start, training_cutoff_date, step="1M", min_train_span="24M", min_fold_rows=50):
    """
    Generate forward-chaining folds.
    
    Rule:
    1. Train on [train_start, fold_start)
    2. Test on [fold_start, fold_end]
       where fold_start >= train_start + min_train_span
    
    If a generated test fold has < min_fold_rows, it is merged into the PREVIOUS fold's test set
    (extending the previous fold's horizon) to avoid tiny calibration samples.
    """
    df["match_date"] = pd.to_datetime(df["match_date"])
    t_start = pd.to_datetime(train_start)
    t_cutoff = pd.to_datetime(training_cutoff_date)
    
    folds = []
    
    # To be safe, let's extract numeric value if it's a string like "24 months"
    def parse_months(m_str):
        if isinstance(m_str, int): return m_str
        s = str(m_str).lower().strip()
        # Handle "1m", "24m", etc.
        if s.endswith('m') and s[:-1].isdigit():
            return int(s[:-1])
        # Handle "1 month", "24 months", etc.
        return int(s.split()[0])

    m_span = parse_months(min_train_span)
    m_step = parse_months(step)

    # Initial start point based on span
    fold_start = t_start + pd.DateOffset(months=m_span)
    
    # ALIGNMENT FIX: Snap to the 1st of the next month if we are doing monthly steps.
    # Users expect "Jan, Feb, Mar" not "Jan 30 - Feb 27".
    is_monthly = "m" in str(step).lower() or "month" in str(step).lower()
    if is_monthly and fold_start.day != 1:
        # Move to the first day of the NEXT calendar month
        # e.g. 2022-06-30 -> 2022-07-01
        fold_start = fold_start + pd.offsets.MonthBegin(1)

    while fold_start <= t_cutoff:
        fold_end = fold_start + pd.DateOffset(months=m_step) - pd.Timedelta(days=1)
        if fold_end > t_cutoff:
            fold_end = t_cutoff
            
        train_df = df[(df["match_date"] >= t_start) & (df["match_date"] < fold_start)].copy()
        test_df = df[(df["match_date"] >= fold_start) & (df["match_date"] <= fold_end)].copy()
        
        # Merge logic
        if len(test_df) < min_fold_rows:
            if folds:
                # Merge with previous fold
                # prev_train stays the same (training set up to prev_start)
                # prev_test is extended from [prev_start, prev_end] to [prev_start, fold_end]
                prev_train, _, prev_start, prev_end = folds.pop()
                
                print(f"  [Fold Merge] Fold starting {fold_start.date()} has {len(test_df)} rows (< {min_fold_rows}). Merging with previous.")
                
                # Re-query the extended test set
                new_test_end = fold_end
                extended_test_df = df[(df["match_date"] >= prev_start) & (df["match_date"] <= new_test_end)].copy()
                
                folds.append((prev_train, extended_test_df, prev_start, new_test_end))
            else:
                # First fold is small. We can't merge back.
                # If it's empty, skip. If it has data, keep it.
                if not test_df.empty:
                    print(f"  [Fold Warning] First fold at {fold_start.date()} is small ({len(test_df)} rows). Keeping.")
                    folds.append((train_df, test_df, fold_start, fold_end))
        else:
            folds.append((train_df, test_df, fold_start, fold_end))
            
        # Optional Sanity Check: Warn if classes are missing in the generated fold
        if folds:
            last_fold_train, last_fold_test = folds[-1][0], folds[-1][1]
            # Assuming 'ft_result_1x2' is the target column in df, but we don't have target col name here.
            # We'll skip strict assertion and just let train.py handle it, 
            # as train.py has the robust class handling now.
            pass

        # Move to next fold
        fold_start = fold_end + pd.Timedelta(days=1)
        
        if fold_start > t_cutoff:
            break
            
    return folds

from sklearn.calibration import IsotonicRegression

class ProbabilityCalibrator:
    """
    A multiclass calibrator supporting 'none' and 'isotonic' methods.
    """
    def __init__(self, method="none"):
        self.method = method
        self.calibrators = {} # outcome_idx -> IsotonicRegression or None
        self.n_classes = 3
        
    def fit(self, p_raw: np.ndarray, y_true: np.ndarray):
        """
        Fit calibrator based on method.
        p_raw: shape (N, 3)
        y_true: shape (N,) with values 0, 1, 2
        """
        if self.method == "none":
            return self
            
        if self.method == "isotonic":
            # Clip raw input for stability
            p_raw = np.clip(p_raw, 1e-6, 1 - 1e-6)
            
            for i in range(self.n_classes):
                y_binary = (y_true == i).astype(int)
                
                # Check for constant class (missing positives or negatives)
                if len(np.unique(y_binary)) < 2:
                    print(f"  Warning: Class {i} is constant in OOF data. Using identity mapping.")
                    self.calibrators[i] = None
                    continue
                
                X = p_raw[:, i] # Isotonic takes 1D array
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
                iso.fit(X, y_binary)
                self.calibrators[i] = iso
                
        return self
        
    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        """
        Apply calibration and renormalize.
        """
        if self.method == "none":
            return p_raw
            
        # Clip input
        p_raw = np.clip(p_raw, 1e-6, 1 - 1e-6)
        p_cal = np.zeros_like(p_raw)
        
        for i in range(self.n_classes):
            X = p_raw[:, i]
            cal = self.calibrators.get(i)
            
            if cal is None:
                # Identity
                p_cal[:, i] = X
            else:
                p_cal[:, i] = cal.predict(X)
        
        # Clip output to avoid pure 0
        p_cal = np.clip(p_cal, 1e-6, 1 - 1e-6)
        
        # Renormalize rows to sum to 1
        row_sums = p_cal.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1.0
        p_cal = p_cal / row_sums
        
        return p_cal
        
    def predict_proba_from_raw(self, p_raw: np.ndarray) -> np.ndarray:
        return self.transform(p_raw)
