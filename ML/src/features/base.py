import numpy as np
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

def ensure_match_datetime(
    df: pd.DataFrame,
    date_col: str = "match_date",
    time_col: str = "match_time",
    out_col: str = "match_datetime",
) -> pd.DataFrame:
    """
    Ensure a match_datetime column exists for stable ordering.
    Uses match_time when available; falls back to match_date at 00:00.
    """
    df = df.copy()
    if out_col in df.columns:
        df[out_col] = pd.to_datetime(df[out_col], errors="coerce")
        return df

    if date_col not in df.columns:
        return df

    match_date = pd.to_datetime(df[date_col], errors="coerce")
    match_date = match_date.dt.normalize()

    if time_col in df.columns:
        time_raw = df[time_col].astype("string").str.strip()
        time_delta = pd.to_timedelta(time_raw, errors="coerce")
        if time_delta.isna().all():
            time_dt = pd.to_datetime(time_raw, errors="coerce")
            seconds = (
                time_dt.dt.hour.fillna(0).astype(int) * 3600
                + time_dt.dt.minute.fillna(0).astype(int) * 60
                + time_dt.dt.second.fillna(0).astype(int)
            )
            time_delta = pd.to_timedelta(seconds, unit="s")
        time_delta = time_delta.fillna(pd.Timedelta(0))
        df[out_col] = match_date + time_delta
    else:
        df[out_col] = match_date

    return df

def infer_season_year(match_date: pd.Series, season_start_month: int = 7) -> pd.Series:
    """
    Infer season year for leagues that span calendar years.
    Default season start month is July (7): Jul-Dec -> current year, Jan-Jun -> previous year.
    """
    dates = pd.to_datetime(match_date, errors="coerce")
    years = dates.dt.year
    months = dates.dt.month
    return years.where(months >= season_start_month, years - 1)

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
    df["season"] = infer_season_year(df["match_date"])
    if "match_id" not in df.columns:
        df["match_id"] = range(len(df))

    df = ensure_match_datetime(df)

    df["league_total_goals"] = (
        pd.to_numeric(df["ft_home_goals"], errors="coerce")
        + pd.to_numeric(df["ft_away_goals"], errors="coerce")
    )

    # Ensure chronological order before expanding stats to avoid leakage.
    sort_cols = ["division", "season", "match_datetime", "match_date", "match_id"] if "match_datetime" in df.columns else ["division", "season", "match_date", "match_id"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

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
