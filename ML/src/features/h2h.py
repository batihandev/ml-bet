import pandas as pd
from .base import _and_int_or_nan

N_H2H = 5

def add_h2h_features(df: pd.DataFrame, n_h2h: int = N_H2H) -> pd.DataFrame:
    """
    Compute head-to-head rolling stats for each home/away pair.
    """
    df = df.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    if "match_id" not in df.columns:
        df["match_id"] = range(len(df))
    df = df.sort_values(["match_date", "match_id"]).reset_index(drop=True)

    df["htd_ft_home_win"] = _and_int_or_nan(df["ht_draw"], df["ft_home_win"])
    df["htd_ft_away_win"] = _and_int_or_nan(df["ht_draw"], df["ft_away_win"])

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

    df = df.groupby(["home_team", "away_team"], group_keys=False).apply(compute_pair, include_groups=True)
    return df
