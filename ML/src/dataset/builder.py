import pandas as pd
from pathlib import Path
from config.allowed_divisions import ALLOWED_DIVISIONS

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

MIN_MATCHES_PER_DIVISION = 300

def _parse_match_date(series: pd.Series) -> pd.Series:
    raw = series.astype("string").str.strip()
    parsed_default = pd.to_datetime(raw, errors="coerce", dayfirst=False)
    parsed_dayfirst = pd.to_datetime(raw, errors="coerce", dayfirst=True)
    count_default = parsed_default.notna().sum()
    count_dayfirst = parsed_dayfirst.notna().sum()

    if count_dayfirst != count_default:
        return parsed_dayfirst if count_dayfirst > count_default else parsed_default

    # Tie-breaker: infer from dd/mm vs mm/dd patterns
    parts = raw.str.extract(r'(?P<a>\d{1,2})[/-](?P<b>\d{1,2})[/-]\d{2,4}')
    a = pd.to_numeric(parts["a"], errors="coerce")
    b = pd.to_numeric(parts["b"], errors="coerce")
    dayfirst_votes = ((a > 12) & (b <= 12)).sum()
    monthfirst_votes = ((b > 12) & (a <= 12)).sum()
    if dayfirst_votes > monthfirst_votes:
        return parsed_dayfirst
    if monthfirst_votes > dayfirst_votes:
        return parsed_default

    return parsed_default

def _parse_match_time(series: pd.Series) -> pd.Series:
    raw = series.astype("string").str.strip()
    time_delta = pd.to_timedelta(raw, errors="coerce")
    if time_delta.isna().all():
        time_dt = pd.to_datetime(raw, errors="coerce")
        seconds = (
            time_dt.dt.hour.fillna(0).astype(int) * 3600
            + time_dt.dt.minute.fillna(0).astype(int) * 60
            + time_dt.dt.second.fillna(0).astype(int)
        )
        time_delta = pd.to_timedelta(seconds, unit="s")
    return time_delta

def load_raw_matches() -> pd.DataFrame:
    matches_path = RAW_DIR / "Matches.csv"
    if not matches_path.exists():
        raise FileNotFoundError(f"Raw file not found: {matches_path}")
    df = pd.read_csv(matches_path, low_memory=False)
    return df

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Division": "division", "MatchDate": "match_date", "MatchTime": "match_time",
        "HomeTeam": "home_team", "AwayTeam": "away_team", "HomeElo": "home_elo", "AwayElo": "away_elo",
        "Form3Home": "form3_home", "Form5Home": "form5_home", "Form3Away": "form3_away", "Form5Away": "form5_away",
        "FTHome": "ft_home_goals", "FTAway": "ft_away_goals", "FTResult": "ft_result",
        "HTHome": "ht_home_goals", "HTAway": "ht_away_goals", "HTResult": "ht_result",
        "HomeShots": "home_shots", "AwayShots": "away_shots",
        "HomeTarget": "home_shots_on_target", "AwayTarget": "away_shots_on_target",
        "HomeFouls": "home_fouls", "AwayFouls": "away_fouls",
        "HomeCorners": "home_corners", "AwayCorners": "away_corners",
        "HomeYellow": "home_yellow", "AwayYellow": "away_yellow",
        "HomeRed": "home_red", "AwayRed": "away_red",
        "OddHome": "odd_home", "OddDraw": "odd_draw", "OddAway": "odd_away",
        "MaxHome": "max_odd_home", "MaxDraw": "max_odd_draw", "MaxAway": "max_odd_away",
        "Over25": "odd_over25", "Under25": "odd_under25",
        "MaxOver25": "max_odd_over25", "MaxUnder25": "max_odd_under25",
        "HandiSize": "handicap_size", "HandiHome": "handicap_home", "HandiAway": "handicap_away",
        "C_LTH": "c_lth", "C_LTA": "c_lta", "C_VHD": "c_vhd", "C_VAD": "c_vad", "C_HTB": "c_htb", "C_PHB": "c_phb",
    }
    df = df.rename(columns=rename_map)
    if "division" in df.columns:
        df["division"] = df["division"].astype("string").str.strip()
    if "match_date" in df.columns:
        df["match_date"] = _parse_match_date(df["match_date"])
        df["match_date"] = df["match_date"].dt.normalize()
    if "match_time" in df.columns:
        df["match_time"] = df["match_time"].astype("string").str.strip()
        time_delta = _parse_match_time(df["match_time"])
    else:
        time_delta = None

    if "match_date" in df.columns:
        if time_delta is not None:
            df["match_datetime"] = df["match_date"] + time_delta.fillna(pd.Timedelta(0))
        else:
            df["match_datetime"] = df["match_date"]

    numeric_cols = [
        "home_elo", "away_elo", "form3_home", "form5_home", "form3_away", "form5_away",
        "ft_home_goals", "ft_away_goals", "ht_home_goals", "ht_away_goals",
        "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target",
        "home_fouls", "away_fouls", "home_corners", "away_corners",
        "home_yellow", "away_yellow", "home_red", "away_red",
        "odd_home", "odd_draw", "odd_away", "max_odd_home", "max_odd_draw", "max_odd_away",
        "odd_over25", "odd_under25", "max_odd_over25", "max_odd_under25",
        "handicap_size", "handicap_home", "handicap_away", "c_lth", "c_lta", "c_vhd", "c_vad", "c_htb", "c_phb",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "match_date" in df.columns:
        sort_cols = ["match_datetime", "match_date"] if "match_datetime" in df.columns else ["match_date"]
        df = df.sort_values(sort_cols)
    return df

def filter_matches(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "match_date" in df.columns:
        df = df[~df["match_date"].isna()]
    if "division" not in df.columns:
        return df
    df["division"] = df["division"].astype("string").str.strip()
    if ALLOWED_DIVISIONS:
        df = df[df["division"].isin(ALLOWED_DIVISIONS)]
    counts = df["division"].value_counts()
    big_divisions = counts[counts >= MIN_MATCHES_PER_DIVISION].index
    df = df[df["division"].isin(big_divisions)]
    return df

def save_processed(df: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = PROCESSED_DIR / "matches.csv"
    parquet_path = PROCESSED_DIR / "matches.parquet"
    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(parquet_path, index=False)
    except:
        pass
