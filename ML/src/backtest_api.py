# backtest_api.py
from typing import Any, Dict
from backtest_ft_1x2 import backtest_ft_1x2_core


def run_backtest_json(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    params expected keys:
      - start_date: str | None
      - end_date: str | None
      - min_edge: float
      - stake: float
      - kelly_mult: float

    Returns JSON-serializable dict:
      {
        "summary": {...},
        "markets": [...],
        "divisions": [...],
        "equity": [...],
      }
    """
    result = backtest_ft_1x2_core(
        start_date=params.get("start_date"),
        end_date=params.get("end_date"),
        min_edge=float(params.get("min_edge", 0.02)),
        stake=float(params.get("stake", 1.0)),
        kelly_mult=float(params.get("kelly_mult", 0.0)),
    )

    summary = result["summary"]
    markets = result["markets"]
    league_stats_df = result["league_stats_df"]
    daily_equity_df = result["daily_equity_df"]

    divisions = (
        league_stats_df.to_dict(orient="records")
        if not league_stats_df.empty
        else []
    )
    equity = (
        daily_equity_df.to_dict(orient="records")
        if not daily_equity_df.empty
        else []
    )

    return {
        "summary": summary,
        "markets": markets,
        "divisions": divisions,
        "equity": equity,
    }
