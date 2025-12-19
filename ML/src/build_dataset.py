import pandas as pd
from pathlib import Path
from config.allowed_divisions import ALLOWED_DIVISIONS

# ---------- paths ----------

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# ---------- filtering config ----------

# Explicit whitelist of divisions to keep.
# You should tune this list based on df["Division"].value_counts()
# from your raw file, but this is a reasonable starting point.


# If you want an extra safety filter inside the allowed set
MIN_MATCHES_PER_DIVISION = 300


def load_raw_matches() -> pd.DataFrame:
    """
    Load the main raw matches file and return a DataFrame.
    For now we use Matches.csv; you can extend this later.
    """
    matches_path = RAW_DIR / "Matches.csv"
    if not matches_path.exists():
        raise FileNotFoundError(f"Raw file not found: {matches_path}")

    # low_memory=False silences that DtypeWarning you saw
    df = pd.read_csv(matches_path, low_memory=False)
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names and types into a canonical schema.
    This matches the sample you sent:
    Division,MatchDate,MatchTime,HomeTeam,AwayTeam,HomeElo,AwayElo,...
    """

    rename_map = {
        "Division": "division",
        "MatchDate": "match_date",
        "MatchTime": "match_time",
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
        "HomeElo": "home_elo",
        "AwayElo": "away_elo",
        "Form3Home": "form3_home",
        "Form5Home": "form5_home",
        "Form3Away": "form3_away",
        "Form5Away": "form5_away",
        "FTHome": "ft_home_goals",
        "FTAway": "ft_away_goals",
        "FTResult": "ft_result",
        "HTHome": "ht_home_goals",
        "HTAway": "ht_away_goals",
        "HTResult": "ht_result",
        "HomeShots": "home_shots",
        "AwayShots": "away_shots",
        "HomeTarget": "home_shots_on_target",
        "AwayTarget": "away_shots_on_target",
        "HomeFouls": "home_fouls",
        "AwayFouls": "away_fouls",
        "HomeCorners": "home_corners",
        "AwayCorners": "away_corners",
        "HomeYellow": "home_yellow",
        "AwayYellow": "away_yellow",
        "HomeRed": "home_red",
        "AwayRed": "away_red",
        "OddHome": "odd_home",
        "OddDraw": "odd_draw",
        "OddAway": "odd_away",
        "MaxHome": "max_odd_home",
        "MaxDraw": "max_odd_draw",
        "MaxAway": "max_odd_away",
        "Over25": "odd_over25",
        "Under25": "odd_under25",
        "MaxOver25": "max_odd_over25",
        "MaxUnder25": "max_odd_under25",
        "HandiSize": "handicap_size",
        "HandiHome": "handicap_home",
        "HandiAway": "handicap_away",
        "C_LTH": "c_lth",
        "C_LTA": "c_lta",
        "C_VHD": "c_vhd",
        "C_VAD": "c_vad",
        "C_HTB": "c_htb",
        "C_PHB": "c_phb",
    }

    df = df.rename(columns=rename_map)

    # make division a clean string
    if "division" in df.columns:
        df["division"] = df["division"].astype("string").str.strip()

    # parse date
    if "match_date" in df.columns:
        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")

    # ensure numeric dtypes where appropriate (ignore parsing errors)
    numeric_cols = [
        "home_elo", "away_elo",
        "form3_home", "form5_home", "form3_away", "form5_away",
        "ft_home_goals", "ft_away_goals",
        "ht_home_goals", "ht_away_goals",
        "home_shots", "away_shots",
        "home_shots_on_target", "away_shots_on_target",
        "home_fouls", "away_fouls",
        "home_corners", "away_corners",
        "home_yellow", "away_yellow",
        "home_red", "away_red",
        "odd_home", "odd_draw", "odd_away",
        "max_odd_home", "max_odd_draw", "max_odd_away",
        "odd_over25", "odd_under25",
        "max_odd_over25", "max_odd_under25",
        "handicap_size", "handicap_home", "handicap_away",
        "c_lth", "c_lta", "c_vhd", "c_vad", "c_htb", "c_phb",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # sort by date for time-based operations later
    if "match_date" in df.columns:
        df = df.sort_values("match_date")

    return df


def filter_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out:
      - rows without a valid date or division
      - any division not in ALLOWED_DIVISIONS
      - (optionally) divisions with too few matches inside that allowed set
    """
    df = df.copy()

    # basic validity
    if "match_date" in df.columns:
        before = len(df)
        df = df[~df["match_date"].isna()]
        print(f"Dropped {before - len(df)} rows with invalid match_date")

    if "division" not in df.columns:
        return df

    df["division"] = df["division"].astype("string").str.strip()

    # explicit allow-list
    if ALLOWED_DIVISIONS:
        before = len(df)
        df = df[df["division"].isin(ALLOWED_DIVISIONS)]
        print(
            f"Kept {len(df)} rows in ALLOWED_DIVISIONS "
            f"({len(set(ALLOWED_DIVISIONS) & set(df['division'].unique()))} divisions present)"
        )

    # optional: drop very small divisions even within allowed set
    counts = df["division"].value_counts()
    big_divisions = counts[counts >= MIN_MATCHES_PER_DIVISION].index
    before = len(df)
    df = df[df["division"].isin(big_divisions)]
    print(
        f"Dropped {before - len(df)} rows from small divisions "
        f"(kept {len(big_divisions)} divisions with >= {MIN_MATCHES_PER_DIVISION} matches)"
    )

    return df


def save_processed(df: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = PROCESSED_DIR / "matches.csv"
    parquet_path = PROCESSED_DIR / "matches.parquet"

    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception as e:
        print(f"Could not save parquet (optional): {e}")

    print(f"Saved processed matches to:\n  {csv_path}")
    if parquet_path.exists():
        print(f"  {parquet_path}")


def main():
    print("Loading raw matches...")
    df_raw = load_raw_matches()
    print(f"Raw shape: {df_raw.shape}")

    print("Normalizing columns and types...")
    df_clean = normalize_columns(df_raw)
    print(f"Clean shape after normalize: {df_clean.shape}")

    if "division" in df_clean.columns:
        print(f"Unique divisions before filter: {df_clean['division'].nunique()}")
        print(df_clean["division"].value_counts().sort_index().head(50))

    print("Filtering matches (allowed divisions + minimum size)...")
    df_filtered = filter_matches(df_clean)
    print(f"Filtered shape: {df_filtered.shape}")
    if "division" in df_filtered.columns:
        print(f"Unique divisions after filter: {df_filtered['division'].nunique()}")
        print(df_filtered["division"].value_counts().sort_index())

    print("Saving processed dataset...")
    save_processed(df_filtered)


if __name__ == "__main__":
    main()
