import argparse
from typing import Optional
from pathlib import Path
import pandas as pd
from .engine import backtest_ft_1x2_core

def backtest_ft_1x2_cli(
    start_date: Optional[str],
    end_date: Optional[str],
    min_edge: float,
    stake: float,
    out_csv: Optional[str],
    out_md: Optional[str],
    kelly_mult: float = 0.0,
) -> None:
    result = backtest_ft_1x2_core(start_date, end_date, min_edge, stake, kelly_mult)
    summary = result["summary"]
    markets_summary = result["markets"]
    league_stats_df = result["league_stats_df"]
    daily = result["daily_equity_df"]
    bets_df = result["bets_df"]

    print(f"Filtered to allowed divisions: {summary['rows_before_division_filter']} -> {summary['rows_after_division_filter']}")
    print(f"Backtesting FT 1X2 from {summary['start_date']} to {summary['end_date']}")
    print(f"Min edge: {summary['min_edge']:.3f}, stake per bet: {summary['stake']}, Kelly multiplier: {summary['kelly_mult']}")

    if summary["total_bets"] == 0:
        print("No bets were placed with the given edge / Kelly settings.")
        return

    print("\n=== FINAL FT 1X2 BACKTEST SUMMARY ===")
    print(f"Total matches in range: {summary['total_matches']}")
    print(f"Total bets: {summary['total_bets']}")
    print(f"Total staked: {summary['total_staked']:.2f}")
    print(f"Total profit: {summary['total_profit']:.2f}")
    print(f"Overall ROI: {summary['overall_roi']:.3f}")

    for m in markets_summary:
        print(f"\n{m['label']}")
        print(f"  Bets: {m['bets']}, Wins: {m['wins']}, Hit rate: {m['hit_rate']:.3f}")
        print(f"  Staked: {m['staked']:.2f}, Profit: {m['profit']:.2f}, ROI: {m['roi']:.3f}")

    if out_csv and not bets_df.empty:
        bets_df.to_csv(out_csv, index=False)
        print(f"\nWrote per-bet FT 1X2 backtest to {out_csv}")

    if out_md:
        md_path = Path(out_md)
        lines = [
            "# FT 1X2 Backtest Summary", "",
            f"- Date range: **{summary['start_date']}** to **{summary['end_date']}**",
            f"- Min edge: **{summary['min_edge']:.3f}**",
            f"- Stake per bet: **{summary['stake']:.2f}**",
            f"- Kelly multiplier: **{summary['kelly_mult']:.2f}** (0 = flat stake)",
            f"- Total matches: **{summary['total_matches']}**",
            f"- Total bets: **{summary['total_bets']}**",
            f"- Total staked: **{summary['total_staked']:.2f}**",
            f"- Total profit: **{summary['total_profit']:.2f}**",
            f"- Overall ROI: **{summary['overall_roi']:.3f}**",
            "", "## Per-market performance", "",
            "| Market | Bets | Wins | Hit rate | Staked | Profit | ROI |",
            "|--------|------|------|----------|--------|--------|-----|",
        ]
        for m in markets_summary:
            lines.append(f"| {m['label']} | {m['bets']} | {m['wins']} | {m['hit_rate']:.3f} | {m['staked']:.2f} | {m['profit']:.2f} | {m['roi']:.3f} |")
        
        md_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Wrote FT 1X2 Markdown summary to {md_path}")

def main():
    parser = argparse.ArgumentParser(description="Backtest FT 1X2 models over a date range.")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--min-edge", type=float, default=0.02)
    parser.add_argument("--stake", type=float, default=1.0)
    parser.add_argument("--kelly-mult", type=float, default=0.0)
    parser.add_argument("--out-csv", type=str, default=None)
    parser.add_argument("--out-md", type=str, default=None)
    args = parser.parse_args()

    backtest_ft_1x2_cli(args.start_date, args.end_date, args.min_edge, args.stake, args.out_csv, args.out_md, args.kelly_mult)

if __name__ == "__main__":
    main()
