import numpy as np
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

def load_processed_matches() -> pd.DataFrame:
    path = PROCESSED_DIR / "matches.csv"
    if not path.exists():
        raise FileNotFoundError(f"Processed matches not found: {path}")
    df = pd.read_csv(path, parse_dates=["match_date"], low_memory=False)
    return df

def _cmp_int_or_nan(a: pd.Series, b: pd.Series, op: str) -> pd.Series:
    """
    Compare a and b and return 1/0, but return NaN if either is NaN.
    This prevents future-match rows (unknown goals) from producing fake targets.
    """
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    valid = a.notna() & b.notna()

    out = pd.Series(np.nan, index=a.index, dtype="float64")
    if op == ">":
        out.loc[valid] = (a.loc[valid] > b.loc[valid]).astype(int)
    elif op == "<":
        out.loc[valid] = (a.loc[valid] < b.loc[valid]).astype(int)
    elif op == "==":
        out.loc[valid] = (a.loc[valid] == b.loc[valid]).astype(int)
    else:
        raise ValueError(f"Unknown op: {op}")
    return out

def _and_int_or_nan(a01: pd.Series, b01: pd.Series) -> pd.Series:
    """
    AND for 0/1 series, but NaN if either is NaN.
    """
    valid = a01.notna() & b01.notna()
    out = pd.Series(np.nan, index=a01.index, dtype="float64")
    out.loc[valid] = ((a01.loc[valid] == 1) & (b01.loc[valid] == 1)).astype(int)
    return out

def add_basic_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # full-time outcomes
    df["ft_home_win"] = _cmp_int_or_nan(df["ft_home_goals"], df["ft_away_goals"], ">")
    df["ft_away_win"] = _cmp_int_or_nan(df["ft_home_goals"], df["ft_away_goals"], "<")
    df["ft_draw"] = _cmp_int_or_nan(df["ft_home_goals"], df["ft_away_goals"], "==")

    # half-time outcomes
    df["ht_home_win"] = _cmp_int_or_nan(df["ht_home_goals"], df["ht_away_goals"], ">")
    df["ht_away_win"] = _cmp_int_or_nan(df["ht_home_goals"], df["ht_away_goals"], "<")
    df["ht_draw"] = _cmp_int_or_nan(df["ht_home_goals"], df["ht_away_goals"], "==")

    # HT Home
    df["target_ht_home_ft_home"] = _and_int_or_nan(df["ht_home_win"], df["ft_home_win"])
    df["target_ht_home_ft_draw"] = _and_int_or_nan(df["ht_home_win"], df["ft_draw"])
    df["target_ht_home_ft_away"] = _and_int_or_nan(df["ht_home_win"], df["ft_away_win"])

    # HT Draw
    df["target_ht_draw_ft_home"] = _and_int_or_nan(df["ht_draw"], df["ft_home_win"])
    df["target_ht_draw_ft_draw"] = _and_int_or_nan(df["ht_draw"], df["ft_draw"])
    df["target_ht_draw_ft_away"] = _and_int_or_nan(df["ht_draw"], df["ft_away_win"])

    # HT Away
    df["target_ht_away_ft_home"] = _and_int_or_nan(df["ht_away_win"], df["ft_home_win"])
    df["target_ht_away_ft_draw"] = _and_int_or_nan(df["ht_away_win"], df["ft_draw"])
    df["target_ht_away_ft_away"] = _and_int_or_nan(df["ht_away_win"], df["ft_away_win"])

    return df

def add_league_context(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df["season"] = df["match_date"].dt.year
    if "match_id" not in df.columns:
        df["match_id"] = range(len(df))

    df["league_total_goals"] = (
        pd.to_numeric(df["ft_home_goals"], errors="coerce")
        + pd.to_numeric(df["ft_away_goals"], errors="coerce")
    )

    # Ensure chronological order before expanding stats to avoid leakage.
    df = df.sort_values(["division", "season", "match_date", "match_id"]).reset_index(drop=True)

    def _expanding_mean_past(s: pd.Series) -> pd.Series:
        # Use only past matches (shifted) to avoid future leakage.
        return s.shift(1).expanding().mean()

    grp = df.groupby(["division", "season"], dropna=False, sort=False)
    df["league_avg_goals"] = (
        grp["league_total_goals"].apply(_expanding_mean_past).reset_index(level=[0, 1], drop=True)
    )
    df["league_home_win_rate"] = (
        grp["ft_home_win"].apply(_expanding_mean_past).reset_index(level=[0, 1], drop=True)
    )
    df["league_draw_rate"] = (
        grp["ft_draw"].apply(_expanding_mean_past).reset_index(level=[0, 1], drop=True)
    )
    df["league_away_win_rate"] = (
        grp["ft_away_win"].apply(_expanding_mean_past).reset_index(level=[0, 1], drop=True)
    )
    return df
