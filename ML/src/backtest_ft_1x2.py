import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import warnings  # for suppressing specific sklearn warning

from config.allowed_divisions import ALLOWED_DIVISIONS
from predict import load_features, load_model, build_X  # reuse core logic

# Suppress only the specific sklearn parallel warning you are seeing
warnings.filterwarnings(
    "ignore",
    message="`sklearn.utils.parallel.delayed` should be used with "
            "`sklearn.utils.parallel.Parallel` to make it possible to "
            "propagate the scikit-learn configuration of the current "
            "thread to the joblib workers.",
    category=UserWarning,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


def load_all_dates(df_all: pd.DataFrame) -> list[date]:
    """Return sorted unique match_date as date objects."""
    return sorted(df_all["match_date"].dt.date.unique())


def _empty_result(
    start_date: str | None,
    end_date: str | None,
    min_edge: float,
    stake: float,
    kelly_mult: float,
    rows_before: int,
    rows_after: int,
) -> dict:
    """
    Helper to build the empty result structure so we do not repeat ourselves.
    This preserves the exact keys and shapes you already use.
    """
    return {
        "summary": {
            "start_date": start_date,
            "end_date": end_date,
            "min_edge": float(min_edge),
            "stake": float(stake),
            "kelly_mult": float(kelly_mult),
            "rows_before_division_filter": int(rows_before),
            "rows_after_division_filter": int(rows_after),
            "total_matches": 0,
            "total_bets": 0,
            "total_staked": 0.0,
            "total_profit": 0.0,
            "overall_roi": 0.0,
        },
        "markets": [],
        "league_stats_df": pd.DataFrame(),
        "daily_equity_df": pd.DataFrame(),
        "bets_df": pd.DataFrame(),
    }


def backtest_ft_1x2_core(
    start_date: str | None,
    end_date: str | None,
    min_edge: float,
    stake: float,
    kelly_mult: float = 0.0,
) -> dict:
    """
    PURE CORE: run FT 1X2 backtest and RETURN results instead of printing.

    Returns a dict with:
      - summary: global stats
      - markets: list of per-market stats
      - league_stats_df: DataFrame (per-division stats)
      - daily_equity_df: DataFrame (per-day equity)
      - bets_df: DataFrame (per-bet logs)
    """
    # ------------------------------------------------------------------
    # Load data and filter to allowed divisions
    # ------------------------------------------------------------------
    df_all = load_features()

    if "division" not in df_all.columns:
        raise ValueError(
            "Column 'division' not found in features.csv; cannot filter by league."
        )

    rows_before = len(df_all)
    df_all = df_all[df_all["division"].isin(ALLOWED_DIVISIONS)].copy()
    rows_after = len(df_all)

    # Precompute normalized implied probabilities terms if odds exist
    if {"odd_home", "odd_draw", "odd_away"}.issubset(df_all.columns):
        df_all["inv_home"] = 1.0 / df_all["odd_home"].where(df_all["odd_home"] > 1.0)
        df_all["inv_draw"] = 1.0 / df_all["odd_draw"].where(df_all["odd_draw"] > 1.0)
        df_all["inv_away"] = 1.0 / df_all["odd_away"].where(df_all["odd_away"] > 1.0)
        df_all["inv_total"] = (
            df_all["inv_home"].fillna(0.0)
            + df_all["inv_draw"].fillna(0.0)
            + df_all["inv_away"].fillna(0.0)
        )
    else:
        df_all["inv_home"] = np.nan
        df_all["inv_draw"] = np.nan
        df_all["inv_away"] = np.nan
        df_all["inv_total"] = np.nan

    all_dates = load_all_dates(df_all)
    if not all_dates:
        return _empty_result(
            start_date=start_date,
            end_date=end_date,
            min_edge=min_edge,
            stake=stake,
            kelly_mult=kelly_mult,
            rows_before=rows_before,
            rows_after=rows_after,
        )

    # ------------------------------------------------------------------
    # Resolve date range
    # ------------------------------------------------------------------
    if start_date is None:
        sd = all_dates[0]
    else:
        sd = pd.to_datetime(start_date).date()

    if end_date is None:
        ed = all_dates[-1]
    else:
        ed = pd.to_datetime(end_date).date()

    date_list = [d for d in all_dates if sd <= d <= ed]
    if not date_list:
        # preserve your existing summary content (stringified sd/ed)
        return _empty_result(
            start_date=str(sd),
            end_date=str(ed),
            min_edge=min_edge,
            stake=stake,
            kelly_mult=kelly_mult,
            rows_before=rows_before,
            rows_after=rows_after,
        )

    # ------------------------------------------------------------------
    # Define FT 1X2 models / markets
    # ------------------------------------------------------------------
    markets = [
        {
            "key": "ft_home_win",
            "label": "FT HOME (1)",
            "model_name": "model_ft_home_win",
            "odds_col": "odd_home",
            "target_col": "ft_home_win",
            "inv_col": "inv_home",
        },
        {
            "key": "ft_draw",
            "label": "FT DRAW (X)",
            "model_name": "model_ft_draw",
            "odds_col": "odd_draw",
            "target_col": "ft_draw",
            "inv_col": "inv_draw",
        },
        {
            "key": "ft_away_win",
            "label": "FT AWAY (2)",
            "model_name": "model_ft_away_win",
            "odds_col": "odd_away",
            "target_col": "ft_away_win",
            "inv_col": "inv_away",
        },
    ]

    # Load models once
    models: dict[str, dict] = {}
    for m in markets:
        key = m["key"]
        model_name = m["model_name"]
        try:
            model, feats = load_model(model_name)
        except FileNotFoundError:
            # skip missing models
            continue

        models[key] = {
            "model": model,
            "features": feats,
            "odds_col": m["odds_col"],
            "target_col": m["target_col"],
            "label": m["label"],
            "inv_col": m["inv_col"],
        }

    if not models:
        # nothing to do – keep same empty structure
        return _empty_result(
            start_date=str(sd),
            end_date=str(ed),
            min_edge=min_edge,
            stake=stake,
            kelly_mult=kelly_mult,
            rows_before=rows_before,
            rows_after=rows_after,
        )

    # ------------------------------------------------------------------
    # Run backtest loop
    # ------------------------------------------------------------------
    total_matches = 0

    bets = {k: 0 for k in models.keys()}
    wins = {k: 0 for k in models.keys()}
    stake_sum = {k: 0.0 for k in models.keys()}
    profit_sum = {k: 0.0 for k in models.keys()}

    total_bets = 0
    total_staked = 0.0
    total_profit = 0.0

    bet_rows: list[dict] = []

    for d in date_list:
        mask_day = df_all["match_date"].dt.date == d
        df_day = df_all[mask_day].copy()
        if df_day.empty:
            continue

        n_matches = len(df_day)
        total_matches += n_matches

        for key, info in models.items():
            model = info["model"]
            feats = info["features"]
            odds_col = info["odds_col"]
            target_col = info["target_col"]
            inv_col = info["inv_col"]
            label = info["label"]

            if odds_col not in df_day.columns or target_col not in df_day.columns:
                continue

            # Model probabilities
            X = build_X(df_day, feats)
            p_raw_all = model.predict_proba(X)[:, 1]
            df_day[f"prob_raw_{key}"] = p_raw_all

            # Iterate over matches of that day for this market
            for _, row in df_day.iterrows():
                p_raw = row[f"prob_raw_{key}"]
                odds = row[odds_col]
                outcome = row[target_col]

                if pd.isna(p_raw) or pd.isna(odds) or odds <= 1.0:
                    continue

                p_hat = float(np.clip(p_raw, 0.001, 0.999))

                inv_total = row.get("inv_total", np.nan)
                if not pd.isna(inv_total) and inv_total > 0:
                    inv_val = row.get(inv_col, np.nan)
                    if pd.isna(inv_val) or inv_val <= 0:
                        p_imp = 1.0 / odds
                    else:
                        p_imp = float(inv_val / inv_total)
                else:
                    p_imp = 1.0 / odds

                edge = p_hat - p_imp
                if edge < min_edge:
                    continue

                # Flat stake or Kelly
                if kelly_mult > 0.0:
                    f_kelly = (p_hat * odds - 1.0) / (odds - 1.0)
                    if f_kelly <= 0:
                        continue
                    bet_size = stake * kelly_mult * f_kelly
                else:
                    bet_size = stake

                # Update counters
                bets[key] += 1
                total_bets += 1
                stake_sum[key] += bet_size
                total_staked += bet_size

                if outcome == 1:
                    profit = bet_size * (odds - 1.0)
                    wins[key] += 1
                else:
                    profit = -bet_size

                profit_sum[key] += profit
                total_profit += profit

                # Log bet row
                bet_rows.append(
                    {
                        "date": d,
                        "division": row.get("division"),
                        "home_team": row.get("home_team"),
                        "away_team": row.get("away_team"),
                        "market": label,
                        "model_key": key,
                        "odds": float(odds),
                        "prob_raw": float(p_raw),
                        "prob_cal": float(p_hat),
                        "prob_implied": float(p_imp),
                        "edge": float(edge),
                        "stake": float(bet_size),
                        "profit": float(profit),
                        "is_win": int(outcome == 1),
                    }
                )

    # If no bets, still return a structured result
    if total_bets == 0:
        summary = {
            "start_date": str(sd),
            "end_date": str(ed),
            "min_edge": float(min_edge),
            "stake": float(stake),
            "kelly_mult": float(kelly_mult),
            "rows_before_division_filter": int(rows_before),
            "rows_after_division_filter": int(rows_after),
            "total_matches": int(total_matches),
            "total_bets": 0,
            "total_staked": 0.0,
            "total_profit": 0.0,
            "overall_roi": 0.0,
        }
        return {
            "summary": summary,
            "markets": [],
            "league_stats_df": pd.DataFrame(),
            "daily_equity_df": pd.DataFrame(),
            "bets_df": pd.DataFrame(),
        }

    bets_df = pd.DataFrame(bet_rows)

    # ------------------------------------------------------------------
    # Per-division stats (league_stats_df)
    # ------------------------------------------------------------------
    if "division" in bets_df.columns:
        league_stats_df = (
            bets_df.groupby("division")
            .agg(
                bets=("division", "size"),
                wins=("is_win", "sum"),
                staked=("stake", "sum"),
                profit=("profit", "sum"),
            )
            .reset_index()
        )
        league_stats_df["hit_rate"] = league_stats_df["wins"] / league_stats_df["bets"]
        league_stats_df["roi"] = league_stats_df["profit"] / league_stats_df["staked"]
    else:
        league_stats_df = pd.DataFrame()

    # ------------------------------------------------------------------
    # Equity curve (daily_equity_df)
    # ------------------------------------------------------------------
    daily_equity_df = (
        bets_df.groupby("date")
        .agg(staked=("stake", "sum"), profit=("profit", "sum"))
        .reset_index()
        .sort_values("date")
    )
    daily_equity_df["cum_profit"] = daily_equity_df["profit"].cumsum()
    daily_equity_df["cum_staked"] = daily_equity_df["staked"].cumsum()
    daily_equity_df["cum_roi"] = (
        daily_equity_df["cum_profit"] / daily_equity_df["cum_staked"]
    )

    # ------------------------------------------------------------------
    # Global market summaries and overall ROI
    # ------------------------------------------------------------------
    overall_roi = total_profit / total_staked if total_staked > 0 else 0.0

    markets_summary = []
    for key, info in models.items():
        st = stake_sum[key]
        pr = profit_sum[key]
        roi = pr / st if st > 0 else 0.0
        hit_rate = wins[key] / bets[key] if bets[key] > 0 else 0.0

        markets_summary.append(
            {
                "key": key,
                "label": info["label"],
                "bets": int(bets[key]),
                "wins": int(wins[key]),
                "staked": float(st),
                "profit": float(pr),
                "roi": float(roi),
                "hit_rate": float(hit_rate),
            }
        )

    summary = {
        "start_date": str(sd),
        "end_date": str(ed),
        "min_edge": float(min_edge),
        "stake": float(stake),
        "kelly_mult": float(kelly_mult),
        "rows_before_division_filter": int(rows_before),
        "rows_after_division_filter": int(rows_after),
        "total_matches": int(total_matches),
        "total_bets": int(total_bets),
        "total_staked": float(total_staked),
        "total_profit": float(total_profit),
        "overall_roi": float(overall_roi),
    }

    return {
        "summary": summary,
        "markets": markets_summary,
        "league_stats_df": league_stats_df,
        "daily_equity_df": daily_equity_df,
        "bets_df": bets_df,
    }


# ----------------------------------------------------------------------
# CLI wrapper that keeps your current behavior (printing, CSV, MD)
# ----------------------------------------------------------------------
def backtest_ft_1x2_cli(
    start_date: str | None,
    end_date: str | None,
    min_edge: float,
    stake: float,
    out_csv: str | None,
    out_md: str | None,
    kelly_mult: float = 0.0,
) -> None:
    result = backtest_ft_1x2_core(
        start_date=start_date,
        end_date=end_date,
        min_edge=min_edge,
        stake=stake,
        kelly_mult=kelly_mult,
    )

    summary = result["summary"]
    markets_summary = result["markets"]
    league_stats_df: pd.DataFrame = result["league_stats_df"]
    daily: pd.DataFrame = result["daily_equity_df"]
    bets_df: pd.DataFrame = result["bets_df"]

    total_matches = summary["total_matches"]
    total_bets = summary["total_bets"]
    total_staked = summary["total_staked"]
    total_profit = summary["total_profit"]
    overall_roi = summary["overall_roi"]

    print(
        f"Filtered to allowed divisions: {summary['rows_before_division_filter']} -> "
        f"{summary['rows_after_division_filter']}"
    )
    print(
        f"Backtesting FT 1X2 from {summary['start_date']} "
        f"to {summary['end_date']}"
    )
    print(
        f"Min edge: {summary['min_edge']:.3f}, stake per bet: {summary['stake']}, "
        f"Kelly multiplier: {summary['kelly_mult']}"
    )

    if total_bets == 0:
        print("No bets were placed with the given edge / Kelly settings.")
        return

    # Final global summary
    print("\n=== FINAL FT 1X2 BACKTEST SUMMARY ===")
    print(f"Total matches in range: {total_matches}")
    print(f"Total bets: {total_bets}")
    print(f"Total staked: {total_staked:.2f}")
    print(f"Total profit: {total_profit:.2f}")
    print(f"Overall ROI: {overall_roi:.3f}")

    for m in markets_summary:
        print(f"\n{m['label']}")
        print(f"  Bets: {m['bets']}, Wins: {m['wins']}, Hit rate: {m['hit_rate']:.3f}")
        print(f"  Staked: {m['staked']:.2f}, Profit: {m['profit']:.2f}, ROI: {m['roi']:.3f}")

    # Per-division markdown (same as before)
    league_md_lines: list[str] = []
    if not league_stats_df.empty:
        league_md_lines.append("## Per-division performance")
        league_md_lines.append("")
        league_md_lines.append("| Division | Bets | Wins | Hit rate | Staked | Profit | ROI |")
        league_md_lines.append("|----------|------|------|----------|--------|--------|-----|")

        for _, row in league_stats_df.sort_values("roi", ascending=False).iterrows():
            league_md_lines.append(
                f"| {row['division']} | {int(row['bets'])} | {int(row['wins'])} | "
                f"{row['hit_rate']:.3f} | {row['staked']:.2f} | {row['profit']:.2f} | "
                f"{row['roi']:.3f} |"
            )

    # Equity curve markdown (first and last 10)
    equity_md_lines: list[str] = []
    if not daily.empty:
        equity_md_lines.append(
            "## Equity curve (per-day cumulative ROI – first and last 10 days)"
        )
        equity_md_lines.append("")
        equity_md_lines.append("| Date | Day stake | Day profit | Cum profit | Cum ROI |")
        equity_md_lines.append("|------|-----------|------------|------------|---------|")

        if len(daily) <= 20:
            rows_to_show = daily.itertuples(index=False)
        else:
            first_part = daily.head(10)
            last_part = daily.tail(10)
            rows_to_show = list(first_part.itertuples(index=False)) + list(
                last_part.itertuples(index=False)
            )

        for r in rows_to_show:
            equity_md_lines.append(
                f"| {r.date} | {r.staked:.2f} | {r.profit:.2f} | "
                f"{r.cum_profit:.2f} | {r.cum_roi:.3f} |"
            )

    # CSV output
    if out_csv and not bets_df.empty:
        out_path = Path(out_csv)
        bets_df.to_csv(out_path, index=False)
        print(f"\nWrote per-bet FT 1X2 backtest to {out_path}")

    # MD output
    if out_md:
        md_path = Path(out_md)
        lines: list[str] = []
        lines.append("# FT 1X2 Backtest Summary")
        lines.append("")
        lines.append(
            f"- Date range: **{summary['start_date']}** to **{summary['end_date']}**"
        )
        lines.append(f"- Min edge: **{summary['min_edge']:.3f}**")
        lines.append(f"- Stake per bet: **{summary['stake']:.2f}**")
        lines.append(f"- Kelly multiplier: **{summary['kelly_mult']:.2f}** (0 = flat stake)")
        lines.append(f"- Total matches: **{total_matches}**")
        lines.append(f"- Total bets: **{total_bets}**")
        lines.append(f"- Total staked: **{total_staked:.2f}**")
        lines.append(f"- Total profit: **{total_profit:.2f}**")
        lines.append(f"- Overall ROI: **{overall_roi:.3f}**")
        lines.append("")
        lines.append("## Per-market performance")
        lines.append("")
        lines.append("| Market | Bets | Wins | Hit rate | Staked | Profit | ROI |")
        lines.append("|--------|------|------|----------|--------|--------|-----|")

        for m in markets_summary:
            lines.append(
                f"| {m['label']} | {m['bets']} | {m['wins']} | {m['hit_rate']:.3f} | "
                f"{m['staked']:.2f} | {m['profit']:.2f} | {m['roi']:.3f} |"
            )

        if league_md_lines:
            lines.append("")
            lines.extend(league_md_lines)

        if equity_md_lines:
            lines.append("")
            lines.extend(equity_md_lines)

        md_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Wrote FT 1X2 Markdown summary to {md_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Backtest FT 1X2 models over a date range (no calibration)."
    )
    parser.add_argument(
        "--start-date", type=str, default=None, help="YYYY-MM-DD (default: first date)"
    )
    parser.add_argument(
        "--end-date", type=str, default=None, help="YYYY-MM-DD (default: last date)"
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=0.02,
        help="Minimum edge (p_model - p_implied) to place a bet (default: 0.02)",
    )
    parser.add_argument(
        "--stake",
        type=float,
        default=1.0,
        help="Base stake per bet (if Kelly multiplier = 0, this is the flat stake).",
    )
    parser.add_argument(
        "--kelly-mult",
        type=float,
        default=0.0,
        help="Kelly multiplier (0 = flat stake, 0.25 = quarter Kelly, 1 = full Kelly).",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Optional path to save per-bet stats as CSV (e.g. backtest_ft_1x2_bets.csv)",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default=None,
        help="Optional path to save Markdown summary (e.g. backtest_ft_1x2_summary.md)",
    )
    args = parser.parse_args()

    backtest_ft_1x2_cli(
        start_date=args.start_date,
        end_date=args.end_date,
        min_edge=args.min_edge,
        stake=args.stake,
        out_csv=args.out_csv,
        out_md=args.out_md,
        kelly_mult=args.kelly_mult,
    )


if __name__ == "__main__":
    main()
