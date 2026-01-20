from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, Any, Dict
import anyio
from backtest.engine import backtest_ft_1x2_core
from ..ws_progress import progress_manager

router = APIRouter(prefix="/backtest", tags=["backtest"])

class BacktestRequest(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    min_edge: float = 0.02
    stake: float = 1.0
    kelly_mult: float = 0.0

def run_backtest_json(params: Dict[str, Any]) -> Dict[str, Any]:
    result = backtest_ft_1x2_core(
        start_date=params.get("start_date"),
        end_date=params.get("end_date"),
        min_edge=float(params.get("min_edge", 0.02)),
        stake=float(params.get("stake", 1.0)),
        kelly_mult=float(params.get("kelly_mult", 0.0)),
    )
    return {
        "summary": result["summary"],
        "markets": result["markets"],
        "divisions": result["league_stats_df"].to_dict(orient="records") if not result["league_stats_df"].empty else [],
        "equity": result["daily_equity_df"].to_dict(orient="records") if not result["daily_equity_df"].empty else [],
    }

@router.post("/ft-1x2")
async def backtest_ft_1x2_api(body: BacktestRequest):
    params = body.dict()
    await progress_manager.broadcast({"type": "backtest_started", "payload": {"job": "ft_1x2", "params": params}})
    try:
        result = await anyio.to_thread.run_sync(run_backtest_json, params)
        await progress_manager.broadcast({"type": "backtest_completed", "payload": {"job": "ft_1x2", "summary": result.get("summary", {})}})
        return result
    except Exception as exc:
        await progress_manager.broadcast({"type": "backtest_failed", "payload": {"job": "ft_1x2", "error": str(exc)}})
        raise
