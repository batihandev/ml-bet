import pandas as pd
from .base import PROCESSED_DIR, load_processed_matches, add_basic_targets, add_league_context
from .history import build_team_history, merge_team_features
from .h2h import add_h2h_features, N_H2H
from .market import add_attack_strength, add_odds_features, add_rule_scores

def build_features(df_matches: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature dataset from an input matches dataframe.
    """
    df = df_matches.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    if "match_id" not in df.columns:
        df["match_id"] = range(len(df))

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

def run_build_features_process():
    print("Loading processed matches...")
    matches = load_processed_matches()
    print(f"Matches shape: {matches.shape}")

    print("Building features (callable pipeline)...")
    features = build_features(matches)
    print(f"Features shape: {features.shape}")

    print("Saving feature dataset...")
    save_features(features)

if __name__ == "__main__":
    run_build_features_process()
