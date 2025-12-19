import argparse
from pathlib import Path

import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"


def load_features() -> pd.DataFrame:
    path = PROCESSED_DIR / "features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")

    df = pd.read_csv(
        path,
        parse_dates=["match_date"],
        low_memory=False,
        dtype={"match_time": "string"},
    )
    return df


def load_model(model_name: str):
    path = MODELS_DIR / f"{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    bundle = joblib.load(path)
    model = bundle["model"]
    features = bundle["features"]
    return model, features


def build_X(df_src: pd.DataFrame, feature_list: list[str]) -> pd.DataFrame:
    X = df_src.copy()

    # Ensure every expected feature column exists
    missing = [c for c in feature_list if c not in X.columns]
    for c in missing:
        X[c] = 0.0

    X = X[feature_list].copy()
    X = X.fillna(0.0)
    return X



def get_predictions_df(pred_date: str | None = None) -> pd.DataFrame:
    """
    Core prediction function.
    Returns a DataFrame for the given date with probability columns added.

    This is the single source of truth used by both:
      - CLI printing in this file
      - any evaluation / API scripts that import it.
    """
    df = load_features()

    if pred_date is None:
        # default to last date in dataset
        date_val = df["match_date"].max().date()
    else:
        date_val = pd.to_datetime(pred_date).date()

    mask = df["match_date"].dt.date == date_val
    df_day = df[mask].copy()

    if df_day.empty:
        raise ValueError(f"No matches found for date {date_val}")

    # ------------------------------------------------------------------
    # 1) HT/FT PATTERN MODELS (existing 9 models)
    #
    # key -> used for probability column name: prob_<key>
    # model_name -> filename in models/ (without .joblib)
    # ------------------------------------------------------------------
    pattern_models = [
        ("ht_home_ft_home", "model_ht_home_ft_home"),
        ("ht_home_ft_draw", "model_ht_home_ft_draw"),
        ("ht_home_ft_away", "model_ht_home_ft_away"),
        ("ht_draw_ft_home", "model_ht_draw_ft_home"),
        ("ht_draw_ft_draw", "model_ht_draw_ft_draw"),
        ("ht_draw_ft_away", "model_ht_draw_ft_away"),
        ("ht_away_ft_home", "model_ht_away_ft_home"),
        ("ht_away_ft_draw", "model_ht_away_ft_draw"),
        ("ht_away_ft_away", "model_ht_away_ft_away"),
    ]

    for key, model_name in pattern_models:
        try:
            model, feats = load_model(model_name)
        except FileNotFoundError:
            # If some HT/FT models are not trained yet, just skip them.
            print(f"[WARN] HT/FT model file not found: {model_name}.joblib – skipping.")
            continue

        X = build_X(df_day, feats)
        proba = model.predict_proba(X)[:, 1]
        df_day[f"prob_{key}"] = proba

    # ------------------------------------------------------------------
    # 2) FT 1X2 MODELS (NEW)
    #
    # We train these as separate binary models:
    #   - ft_home_win  -> model_ft_home_win
    #   - ft_draw      -> model_ft_draw
    #   - ft_away_win  -> model_ft_away_win
    #
    # We add probability columns:
    #   - prob_ft_home_win
    #   - prob_ft_draw
    #   - prob_ft_away_win
    #
    # These are raw 0..1 probabilities from each binary classifier.
    # ------------------------------------------------------------------
    ft_models = [
        ("ft_home_win", "model_ft_home_win"),
        ("ft_draw", "model_ft_draw"),
        ("ft_away_win", "model_ft_away_win"),
    ]

    for outcome_key, model_name in ft_models:
        try:
            model, feats = load_model(model_name)
        except FileNotFoundError:
            print(f"[WARN] FT model file not found: {model_name}.joblib – skipping.")
            continue

        X = build_X(df_day, feats)
        proba = model.predict_proba(X)[:, 1]
        df_day[f"prob_{outcome_key}"] = proba

    return df_day


def print_predictions(df_day: pd.DataFrame) -> None:
    """
    Pretty-print top candidates from a predictions DataFrame.
    """
    date_val = df_day["match_date"].iloc[0].date()
    print(f"Found {len(df_day)} matches for {date_val}")

    base_cols = [
        "division",
        "match_date",
        "home_team",
        "away_team",
        "odd_home",
        "odd_draw",
        "odd_away",
        "rule_score_home_htdftw",
        "rule_score_away_htdftw",
    ]
    base_cols = [c for c in base_cols if c in df_day.columns]

    # ------------------------------------------------------------------
    # A) HT/FT patterns (existing behavior, unchanged)
    # ------------------------------------------------------------------
    pattern_labels = [
        ("ht_home_ft_home", "HT HOME win + FT HOME win"),
        ("ht_home_ft_draw", "HT HOME win + FT DRAW"),
        ("ht_home_ft_away", "HT HOME win + FT AWAY win"),
        ("ht_draw_ft_home", "HT DRAW + FT HOME win"),
        ("ht_draw_ft_draw", "HT DRAW + FT DRAW"),
        ("ht_draw_ft_away", "HT DRAW + FT AWAY win"),
        ("ht_away_ft_home", "HT AWAY win + FT HOME win"),
        ("ht_away_ft_draw", "HT AWAY win + FT DRAW"),
        ("ht_away_ft_away", "HT AWAY win + FT AWAY win"),
    ]

    for key, label in pattern_labels:
        prob_col = f"prob_{key}"
        if prob_col not in df_day.columns:
            continue

        show_cols = base_cols + [prob_col]
        print(f"\nTop candidates for {label}:")
        print(
            df_day.sort_values(prob_col, ascending=False)[show_cols]
            .head(20)
            .to_string(index=False)
        )

    # ------------------------------------------------------------------
    # B) FT 1X2 summary (NEW, optional)
    #
    # If FT probability columns exist, show a quick table of top matches
    # ranked by the strongest FT outcome probability.
    # ------------------------------------------------------------------
    ft_cols = ["prob_ft_home_win", "prob_ft_draw", "prob_ft_away_win"]
    if all(c in df_day.columns for c in ft_cols):
        df_ft = df_day.copy()
        df_ft["prob_ft_max"] = df_ft[ft_cols].max(axis=1)

        show_cols_ft = base_cols + ft_cols + ["prob_ft_max"]
        show_cols_ft = [c for c in show_cols_ft if c in df_ft.columns]

        print("\nTop matches by max FT 1X2 model probability:")
        print(
            df_ft.sort_values("prob_ft_max", ascending=False)[show_cols_ft]
            .head(40)
            .to_string(index=False)
        )


def main():
    parser = argparse.ArgumentParser(description="Predict HT/FT patterns (and FT 1X2) for a given date.")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Prediction date in YYYY-MM-DD (defaults to last date in dataset).",
    )
    args = parser.parse_args()

    df_day = get_predictions_df(args.date)
    print_predictions(df_day)


if __name__ == "__main__":
    main()
