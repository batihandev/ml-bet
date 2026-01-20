# ML/src/features.py
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# Horizons for team form features (number of recent matches)
FORM_WINDOWS = [3, 5, 10]   # short / medium / longer recent form
N_H2H = 5                   # number of recent head-to-head matches for H2H features


def load_processed_matches() -> pd.DataFrame:
    path = PROCESSED_DIR / "matches.csv"
    if not path.exists():
        raise FileNotFoundError(f"Processed matches not found: {path}")
    df = pd.read_csv(path, parse_dates=["match_date"], low_memory=False)
    return df


# ----------------------------------------------------------------------
# Helpers: safe targets with NaNs (for synthetic future matches)
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# 0) Basic targets (SAFE WITH NaNs)
# ----------------------------------------------------------------------
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

    # 9 separate HT/FT pattern targets
    # Naming: target_ht_<H/D/A>_ft_<H/D/A>
    # H = home win, D = draw, A = away win

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


# ----------------------------------------------------------------------
# 1) League-season context (FIXED + NaN-safe)
# ----------------------------------------------------------------------
def add_league_context(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["season"] = pd.to_datetime(df["match_date"], errors="coerce").dt.year  # simple season definition

    # Compute total goals once (NaN-safe)
    df["league_total_goals"] = (
        pd.to_numeric(df["ft_home_goals"], errors="coerce")
        + pd.to_numeric(df["ft_away_goals"], errors="coerce")
    )

    league_stats = (
        df.groupby(["division", "season"], dropna=False)
        .agg(
            league_avg_goals=("league_total_goals", "mean"),
            league_home_win_rate=("ft_home_win", "mean"),
            league_draw_rate=("ft_draw", "mean"),
            league_away_win_rate=("ft_away_win", "mean"),
        )
        .reset_index()
    )

    df = df.merge(league_stats, on=["division", "season"], how="left")
    return df


# ----------------------------------------------------------------------
# 2) Team history (form + rest + congestion + home/away-specific form)
# ----------------------------------------------------------------------
def build_team_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-team rolling form features for multiple horizons.

    For each team and each match, we compute for horizons in FORM_WINDOWS (3,5,10)
    plus an "all-time" expanding window (formAll_*), all from the team perspective.

    Metrics per horizon:
      - ht_draw_rate
      - ft_win_rate
      - ft_draw_rate
      - htd_ftw_rate   (HT draw + FT win)
      - goals_for_avg
      - goals_against_avg

    Additional:
      - days_since_last
      - matches_last_7d / 14d / 21d (fixture congestion)
      - home-only form (team as home), away-only form (team as away)
    """
    df = df.copy()
    if "match_id" not in df.columns:
        df["match_id"] = range(len(df))

    # Ensure datetime
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")

    # home rows
    home = df[
        [
            "match_id",
            "match_date",
            "home_team",
            "ht_home_goals",
            "ht_away_goals",
            "ft_home_goals",
            "ft_away_goals",
        ]
    ].copy()
    home["team"] = home["home_team"]
    home["is_home"] = 1
    home["gf_ht"] = home["ht_home_goals"]
    home["ga_ht"] = home["ht_away_goals"]
    home["gf_ft"] = home["ft_home_goals"]
    home["ga_ft"] = home["ft_away_goals"]

    # away rows
    away = df[
        [
            "match_id",
            "match_date",
            "away_team",
            "ht_home_goals",
            "ht_away_goals",
            "ft_home_goals",
            "ft_away_goals",
        ]
    ].copy()
    away["team"] = away["away_team"]
    away["is_home"] = 0
    away["gf_ht"] = away["ht_away_goals"]
    away["ga_ht"] = away["ht_home_goals"]
    away["gf_ft"] = away["ft_away_goals"]
    away["ga_ft"] = away["ft_home_goals"]

    team_matches = pd.concat([home, away], ignore_index=True)

    # Coerce to numeric for safety
    for col in ["gf_ht", "ga_ht", "gf_ft", "ga_ft"]:
        team_matches[col] = pd.to_numeric(team_matches[col], errors="coerce")

    # indicators from team perspective
    team_matches["ht_draw"] = _cmp_int_or_nan(team_matches["gf_ht"], team_matches["ga_ht"], "==")
    team_matches["ft_win"] = _cmp_int_or_nan(team_matches["gf_ft"], team_matches["ga_ft"], ">")
    team_matches["ft_draw_flag"] = _cmp_int_or_nan(team_matches["gf_ft"], team_matches["ga_ft"], "==")
    team_matches["htd_ft_win"] = _and_int_or_nan(team_matches["ht_draw"], team_matches["ft_win"])

    team_matches = team_matches.sort_values(["team", "match_date", "match_id"]).reset_index(drop=True)

    def compute_rolling(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values(["match_date", "match_id"]).reset_index(drop=True)

        # Rest: days since last match (up to previous match)
        group["days_since_last"] = group["match_date"].diff().dt.days.shift(1)

        # Fixture congestion: matches in last X days (excluding current)
        dates = group["match_date"].values
        for window_days, colname in [
            (7, "matches_last_7d"),
            (14, "matches_last_14d"),
            (21, "matches_last_21d"),
        ]:
            window = np.timedelta64(window_days, "D")
            counts = []
            start = 0
            for i in range(len(dates)):
                while dates[i] - dates[start] > window:
                    start += 1
                counts.append(i - start)  # strictly before current within window
            group[colname] = counts

        # finite rolling windows (3,5,10) - overall form
        for w in FORM_WINDOWS:
            prefix = f"form{w}"
            group[f"{prefix}_ht_draw_rate"] = group["ht_draw"].shift(1).rolling(w).mean()
            group[f"{prefix}_ft_win_rate"] = group["ft_win"].shift(1).rolling(w).mean()
            group[f"{prefix}_ft_draw_rate"] = group["ft_draw_flag"].shift(1).rolling(w).mean()
            group[f"{prefix}_htd_ftw_rate"] = group["htd_ft_win"].shift(1).rolling(w).mean()
            group[f"{prefix}_goals_for_avg"] = group["gf_ft"].shift(1).rolling(w).mean()
            group[f"{prefix}_goals_against_avg"] = group["ga_ft"].shift(1).rolling(w).mean()

        # all-time expanding window (up to previous match)
        group["formAll_ht_draw_rate"] = group["ht_draw"].shift(1).expanding().mean()
        group["formAll_ft_win_rate"] = group["ft_win"].shift(1).expanding().mean()
        group["formAll_ft_draw_rate"] = group["ft_draw_flag"].shift(1).expanding().mean()
        group["formAll_htd_ftw_rate"] = group["htd_ft_win"].shift(1).expanding().mean()
        group["formAll_goals_for_avg"] = group["gf_ft"].shift(1).expanding().mean()
        group["formAll_goals_against_avg"] = group["ga_ft"].shift(1).expanding().mean()

        # Home-only and away-only form (venue-specific)
        # Use index-aligned assignment instead of merge loops (faster and safer).
        # Home-only and away-only form (venue-specific) WITHOUT merges (prevents duplicates)
        for w in FORM_WINDOWS:
            col_home = f"form{w}_home_only_ft_win_rate"
            col_away = f"form{w}_away_only_ft_win_rate"

            group[col_home] = np.nan
            group[col_away] = np.nan

            mask_home = group["is_home"] == 1
            if mask_home.any():
                sub = group.loc[mask_home, ["match_date", "match_id", "ft_win"]].copy()
                sub = sub.sort_values(["match_date", "match_id"])
                vals = sub["ft_win"].shift(1).rolling(w).mean().to_numpy()
                group.loc[mask_home, col_home] = vals

            mask_away = group["is_home"] == 0
            if mask_away.any():
                sub = group.loc[mask_away, ["match_date", "match_id", "ft_win"]].copy()
                sub = sub.sort_values(["match_date", "match_id"])
                vals = sub["ft_win"].shift(1).rolling(w).mean().to_numpy()
                group.loc[mask_away, col_away] = vals

        return group

    team_matches = team_matches.groupby("team", group_keys=False).apply(compute_rolling)
    return team_matches


# ----------------------------------------------------------------------
# 3) Merge team-level features back to match rows
# ----------------------------------------------------------------------
def merge_team_features(matches: pd.DataFrame, team_matches: pd.DataFrame) -> pd.DataFrame:
    matches = matches.copy()
    tm = team_matches.copy()

    if "match_id" not in matches.columns:
        matches["match_id"] = range(len(matches))

    base_cols = ["match_id", "team", "is_home"]
    form_cols = [c for c in tm.columns if c.startswith("form")]
    congestion_cols = ["days_since_last", "matches_last_7d", "matches_last_14d", "matches_last_21d"]

    feature_cols = base_cols + form_cols + [c for c in congestion_cols if c in tm.columns]
    tm = tm[feature_cols].copy()

    # Home features: is_home == 1
    home_feat = tm[tm["is_home"] == 1].copy()
    home_feat = home_feat.drop(columns=["team", "is_home"])
    # CRITICAL: ensure unique match_id to avoid merge blow-up
    home_feat = home_feat.drop_duplicates(subset=["match_id"], keep="first")
    home_feat = home_feat.rename(columns=lambda c: c if c == "match_id" else f"home_{c}")

    # Away features: is_home == 0
    away_feat = tm[tm["is_home"] == 0].copy()
    away_feat = away_feat.drop(columns=["team", "is_home"])
    # CRITICAL: ensure unique match_id to avoid merge blow-up
    away_feat = away_feat.drop_duplicates(subset=["match_id"], keep="first")
    away_feat = away_feat.rename(columns=lambda c: c if c == "match_id" else f"away_{c}")

    merged = matches.merge(home_feat, on="match_id", how="left")
    merged = merged.merge(away_feat, on="match_id", how="left")
    return merged

def add_attack_strength(df: pd.DataFrame) -> pd.DataFrame:
    """
    Leakage-safe attack strength features.

    home_attack_strength:
      (home team's prior avg goals scored) / (league prior avg home goals)

    away_attack_strength:
      (away team's prior avg goals scored) / (league prior avg away goals)

    Computed within (division, season) to avoid cross-league mixing.
    """
    df = df.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    if "season" not in df.columns:
        df["season"] = df["match_date"].dt.year

    # Ensure numeric
    df["ft_home_goals"] = pd.to_numeric(df["ft_home_goals"], errors="coerce")
    df["ft_away_goals"] = pd.to_numeric(df["ft_away_goals"], errors="coerce")

    # Stable ordering
    if "match_id" not in df.columns:
        df["match_id"] = range(len(df))
    df = df.sort_values(["division", "season", "match_date", "match_id"]).reset_index(drop=True)

    # League prior averages (within division+season)
    df["league_avg_home_goals_prev"] = (
        df.groupby(["division", "season"])["ft_home_goals"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=[0, 1], drop=True)
    )
    df["league_avg_away_goals_prev"] = (
        df.groupby(["division", "season"])["ft_away_goals"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=[0, 1], drop=True)
    )

    # Team prior scoring averages (scored goals in prior matches), within division+season
    df["home_scored_prev_avg"] = (
        df.groupby(["division", "season", "home_team"])["ft_home_goals"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=[0, 1, 2], drop=True)
    )
    df["away_scored_prev_avg"] = (
        df.groupby(["division", "season", "away_team"])["ft_away_goals"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=[0, 1, 2], drop=True)
    )

    # Strength ratios
    df["home_attack_strength"] = df["home_scored_prev_avg"] / df["league_avg_home_goals_prev"].where(
        df["league_avg_home_goals_prev"] > 0
    )
    df["away_attack_strength"] = df["away_scored_prev_avg"] / df["league_avg_away_goals_prev"].where(
        df["league_avg_away_goals_prev"] > 0
    )

    # Optional: clean up intermediate columns (keep if you want debug)
    df = df.drop(
        columns=[
            "league_avg_home_goals_prev",
            "league_avg_away_goals_prev",
            "home_scored_prev_avg",
            "away_scored_prev_avg",
        ],
        errors="ignore",
    )

    return df


# ----------------------------------------------------------------------
# 4) Richer H2H features (SAFE WITH NaNs)
# ----------------------------------------------------------------------
def add_h2h_features(df: pd.DataFrame, n_h2h: int = N_H2H) -> pd.DataFrame:
    """
    Compute head-to-head rolling stats for each home/away pair
    from the home team's perspective.
    """
    df = df.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.sort_values(["match_date", "match_id"]).reset_index(drop=True)

    # pattern flags from home perspective (NaN-safe via helper)
    df["htd_ft_home_win"] = _and_int_or_nan(df["ht_draw"], df["ft_home_win"])
    df["htd_ft_away_win"] = _and_int_or_nan(df["ht_draw"], df["ft_away_win"])

    # generic H2H stats (NaN-safe)
    ft_hg = pd.to_numeric(df["ft_home_goals"], errors="coerce")
    ft_ag = pd.to_numeric(df["ft_away_goals"], errors="coerce")

    df["btts"] = _and_int_or_nan((ft_hg > 0).astype("float64"), (ft_ag > 0).astype("float64"))
    df["total_goals"] = ft_hg + ft_ag

    def compute_pair(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values(["match_date", "match_id"]).reset_index(drop=True)

        group["h2h_htd_ft_home_win_rate"] = group["htd_ft_home_win"].shift(1).rolling(n_h2h).mean()
        group["h2h_htd_ft_away_win_rate"] = group["htd_ft_away_win"].shift(1).rolling(n_h2h).mean()
        group["h2h_matches_count"] = group["htd_ft_home_win"].shift(1).rolling(n_h2h).count()

        group["h2h_total_goals_avg"] = group["total_goals"].shift(1).rolling(n_h2h).mean()
        group["h2h_btts_rate"] = group["btts"].shift(1).rolling(n_h2h).mean()
        group["h2h_home_win_rate"] = group["ft_home_win"].shift(1).rolling(n_h2h).mean()
        group["h2h_away_win_rate"] = group["ft_away_win"].shift(1).rolling(n_h2h).mean()
        return group

    df = df.groupby(["home_team", "away_team"], group_keys=False).apply(compute_pair)
    return df


# ----------------------------------------------------------------------
# 5) Rule scores (your original "rule strength" idea)
# ----------------------------------------------------------------------
def add_rule_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["rule_score_home_htdftw"] = df[
        ["home_form5_htd_ftw_rate", "away_form5_htd_ftw_rate", "h2h_htd_ft_home_win_rate"]
    ].mean(axis=1)

    df["rule_score_away_htdftw"] = df[
        ["home_form5_htd_ftw_rate", "away_form5_htd_ftw_rate", "h2h_htd_ft_away_win_rate"]
    ].mean(axis=1)

    df["passes_rule_home_htdftw"] = (
        (df["home_form5_htd_ftw_rate"] >= 0.51)
        & (df["away_form5_htd_ftw_rate"] >= 0.51)
        & (df["h2h_htd_ft_home_win_rate"] >= 0.51)
    ).astype(int)

    df["passes_rule_away_htdftw"] = (
        (df["home_form5_htd_ftw_rate"] >= 0.51)
        & (df["away_form5_htd_ftw_rate"] >= 0.51)
        & (df["h2h_htd_ft_away_win_rate"] >= 0.51)
    ).astype(int)

    return df


# ----------------------------------------------------------------------
# 6) Odds-derived structural features
# ----------------------------------------------------------------------
def add_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if {"odd_home", "odd_draw", "odd_away"}.issubset(df.columns):
        df["inv_home"] = 1.0 / df["odd_home"].where(df["odd_home"] > 1.0)
        df["inv_draw"] = 1.0 / df["odd_draw"].where(df["odd_draw"] > 1.0)
        df["inv_away"] = 1.0 / df["odd_away"].where(df["odd_away"] > 1.0)

        df["inv_total"] = (
            df["inv_home"].fillna(0.0)
            + df["inv_draw"].fillna(0.0)
            + df["inv_away"].fillna(0.0)
        )

        df["p_home_imp_norm"] = df["inv_home"] / df["inv_total"].where(df["inv_total"] > 0)
        df["p_draw_imp_norm"] = df["inv_draw"] / df["inv_total"].where(df["inv_total"] > 0)
        df["p_away_imp_norm"] = df["inv_away"] / df["inv_total"].where(df["inv_total"] > 0)

        df["fav_prob_norm"] = df[["p_home_imp_norm", "p_away_imp_norm"]].max(axis=1)
        df["market_balance"] = (df["p_home_imp_norm"] - df["p_away_imp_norm"]).abs()
    else:
        df["inv_home"] = np.nan
        df["inv_draw"] = np.nan
        df["inv_away"] = np.nan
        df["inv_total"] = np.nan
        df["p_home_imp_norm"] = np.nan
        df["p_draw_imp_norm"] = np.nan
        df["p_away_imp_norm"] = np.nan
        df["fav_prob_norm"] = np.nan
        df["market_balance"] = np.nan

    return df


# ----------------------------------------------------------------------
# Callable pipeline (IMPORTANT for live-with-history)
# ----------------------------------------------------------------------
def build_features(df_matches: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature dataset from an input matches dataframe.
    Works for:
      - full historical batch
      - historical + synthetic future match (goals are NaN)
    """
    df = df_matches.copy()

    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")

    if "match_id" not in df.columns:
        df["match_id"] = range(len(df))

    # Stable global ordering for rolling computations
    df = df.sort_values(["match_date", "match_id"]).reset_index(drop=True)

    df = add_basic_targets(df)
    df = add_league_context(df)
    df = add_attack_strength(df)
    team_hist = build_team_history(df)
    df = merge_team_features(df, team_hist)

    df = add_odds_features(df)
    df = add_h2h_features(df, n_h2h=N_H2H)
    df = add_rule_scores(df)

    return df


# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------
def save_features(df: pd.DataFrame) -> None:
    path_csv = PROCESSED_DIR / "features.csv"
    path_parquet = PROCESSED_DIR / "features.parquet"

    df.to_csv(path_csv, index=False)
    try:
        df.to_parquet(path_parquet, index=False)
    except Exception as e:
        print(f"Could not save parquet (optional): {e}")

    print(f"Saved features to:\n  {path_csv}")
    if path_parquet.exists():
        print(f"  {path_parquet}")


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------
def run_build_features_process():
    print("Loading processed matches...")
    matches = load_processed_matches()
    print(f"Matches shape: {matches.shape}")

    print("Building features (callable pipeline)...")
    features = build_features(matches)
    print(f"Features shape: {features.shape}")

    print("Saving feature dataset...")
    save_features(features)

def main():
    run_build_features_process()


if __name__ == "__main__":
    main()
