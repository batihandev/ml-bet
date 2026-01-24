import numpy as np
import pandas as pd
from .base import _cmp_int_or_nan, _and_int_or_nan

FORM_WINDOWS = [3, 5, 10]

def build_team_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-team rolling form features for multiple horizons.
    """
    df = df.copy()
    if "match_id" not in df.columns:
        df["match_id"] = range(len(df))

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

    for col in ["gf_ht", "ga_ht", "gf_ft", "ga_ft"]:
        team_matches[col] = pd.to_numeric(team_matches[col], errors="coerce")

    team_matches["ht_draw"] = _cmp_int_or_nan(team_matches["gf_ht"], team_matches["ga_ht"], "==")
    team_matches["ft_win"] = _cmp_int_or_nan(team_matches["gf_ft"], team_matches["ga_ft"], ">")
    team_matches["ft_draw_flag"] = _cmp_int_or_nan(team_matches["gf_ft"], team_matches["ga_ft"], "==")
    team_matches["htd_ft_win"] = _and_int_or_nan(team_matches["ht_draw"], team_matches["ft_win"])

    team_matches = team_matches.sort_values(["team", "match_date", "match_id"]).reset_index(drop=True)

    def compute_rolling(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values(["match_date", "match_id"]).reset_index(drop=True)
        if "team" not in group.columns:
            group["team"] = group.name
        group["days_since_last"] = group["match_date"].diff().dt.days.shift(1)

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
                counts.append(i - start)
            group[colname] = counts

        for w in FORM_WINDOWS:
            prefix = f"form{w}"
            group[f"{prefix}_ht_draw_rate"] = group["ht_draw"].shift(1).rolling(w).mean()
            group[f"{prefix}_ft_win_rate"] = group["ft_win"].shift(1).rolling(w).mean()
            group[f"{prefix}_ft_draw_rate"] = group["ft_draw_flag"].shift(1).rolling(w).mean()
            group[f"{prefix}_htd_ftw_rate"] = group["htd_ft_win"].shift(1).rolling(w).mean()
            group[f"{prefix}_goals_for_avg"] = group["gf_ft"].shift(1).rolling(w).mean()
            group[f"{prefix}_goals_against_avg"] = group["ga_ft"].shift(1).rolling(w).mean()

        group["formAll_ht_draw_rate"] = group["ht_draw"].shift(1).expanding().mean()
        group["formAll_ft_win_rate"] = group["ft_win"].shift(1).expanding().mean()
        group["formAll_ft_draw_rate"] = group["ft_draw_flag"].shift(1).expanding().mean()
        group["formAll_htd_ftw_rate"] = group["htd_ft_win"].shift(1).expanding().mean()
        group["formAll_goals_for_avg"] = group["gf_ft"].shift(1).expanding().mean()
        group["formAll_goals_against_avg"] = group["ga_ft"].shift(1).expanding().mean()

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

    try:
        team_matches = team_matches.groupby("team", group_keys=False).apply(compute_rolling, include_groups=False)
    except TypeError:
        # Older pandas versions don't support include_groups.
        team_matches = team_matches.groupby("team", group_keys=False).apply(compute_rolling)
    return team_matches

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

    home_feat = tm[tm["is_home"] == 1].copy()
    home_feat = home_feat.drop(columns=["team", "is_home"])
    home_feat = home_feat.drop_duplicates(subset=["match_id"], keep="first")
    home_feat = home_feat.rename(columns=lambda c: c if c == "match_id" else f"home_{c}")

    away_feat = tm[tm["is_home"] == 0].copy()
    away_feat = away_feat.drop(columns=["team", "is_home"])
    away_feat = away_feat.drop_duplicates(subset=["match_id"], keep="first")
    away_feat = away_feat.rename(columns=lambda c: c if c == "match_id" else f"away_{c}")

    merged = matches.merge(home_feat, on="match_id", how="left")
    merged = merged.merge(away_feat, on="match_id", how="left")
    return merged
