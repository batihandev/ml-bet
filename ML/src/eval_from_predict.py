import argparse
from predict import get_predictions_df  # reuse core prediction logic


def eval_from_predictions(pred_date: str, thr: float):
    df_day = get_predictions_df(pred_date)

    # Define all 9 patterns: key (suffix used in prob col), target col, label
    patterns = [
        ("ht_home_ft_home", "target_ht_home_ft_home", "HT HOME win + FT HOME win"),
        ("ht_home_ft_draw", "target_ht_home_ft_draw", "HT HOME win + FT DRAW"),
        ("ht_home_ft_away", "target_ht_home_ft_away", "HT HOME win + FT AWAY win"),
        ("ht_draw_ft_home", "target_ht_draw_ft_home", "HT DRAW + FT HOME win"),
        ("ht_draw_ft_draw", "target_ht_draw_ft_draw", "HT DRAW + FT DRAW"),
        ("ht_draw_ft_away", "target_ht_draw_ft_away", "HT DRAW + FT AWAY win"),
        ("ht_away_ft_home", "target_ht_away_ft_home", "HT AWAY win + FT HOME win"),
        ("ht_away_ft_draw", "target_ht_away_ft_draw", "HT AWAY win + FT DRAW"),
        ("ht_away_ft_away", "target_ht_away_ft_away", "HT AWAY win + FT AWAY win"),
    ]

    def summarize(target_col: str, prob_col: str, pick_col: str, name: str):
        picked = df_day[df_day[pick_col] == 1]
        total_picked = len(picked)
        hits = picked[target_col].sum() if total_picked > 0 else 0
        hit_rate = hits / total_picked if total_picked > 0 else 0.0
        base_rate = df_day[target_col].mean()

        print(f"\n=== {name} ===")
        print(f"Total matches: {len(df_day)}")
        print(f"Base rate (all matches): {base_rate:.4f}")
        print(f"Picked {total_picked} matches at thr={thr:.2f}")
        print(f"Hits among picked: {hits}  (hit rate: {hit_rate:.4f})")

        if total_picked > 0:
            cols = [
                "division",
                "match_date",
                "home_team",
                "away_team",
                "odd_home",
                "odd_draw",
                "odd_away",
                "rule_score_home_htdftw",
                "rule_score_away_htdftw",
                prob_col,
                target_col,
                pick_col,
                "ht_result",
                "ft_result",
            ]
            cols = [c for c in cols if c in df_day.columns]
            print("\nPicked matches (top 20 by probability):")
            print(
                picked.sort_values(prob_col, ascending=False)[cols]
                .head(20)
                .to_string(index=False)
            )

    # Add pick flags and summarize for each pattern
    for key, target_col, label in patterns:
        prob_col = f"prob_{key}"
        pick_col = f"pick_{key}"

        if prob_col not in df_day.columns or target_col not in df_day.columns:
            print(f"\n[WARN] Missing columns for {label}, skipping.")
            continue

        df_day[pick_col] = (df_day[prob_col] >= thr).astype(int)
        summarize(target_col, prob_col, pick_col, label)


def main():
    parser = argparse.ArgumentParser(description="Evaluate all 9 HT/FT patterns for a given date.")
    parser.add_argument("--date", type=str, required=True, help="Date to evaluate, YYYY-MM-DD")
    parser.add_argument("--thr", type=float, default=0.30, help="Probability threshold (default 0.30)")
    args = parser.parse_args()

    eval_from_predictions(args.date, args.thr)


if __name__ == "__main__":
    main()
