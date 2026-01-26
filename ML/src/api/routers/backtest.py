from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, Any, Dict, List, Tuple
import anyio
from production.backtest import backtest_production_1x2 as backtest_ft_1x2_core
from production.backtest_sweep import run_backtest_sweep
from ..ws_progress import progress_manager

router = APIRouter(prefix="/backtest", tags=["backtest"])

class BacktestRequest(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    min_edge: float = 0.05
    min_ev: float = 0.0
    stake: float = 1.0
    kelly_mult: float = 0.0
    selection_mode: str = "best_ev"
    blend_alpha: float = 1.0
    debug: int = 0

class SweepRequest(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    edge_range: Tuple[float, float, float] = (0.0, 0.10, 0.01)
    ev_range: Tuple[float, float, float] = (0.0, 0.10, 0.01)
    alpha_range: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    stake: float = 1.0
    kelly_mult: float = 0.0
    min_bets: int = 300
    bootstrap_n: int = 1000
    selection_mode: str = "best_ev"
    debug: int = 0

def run_backtest_json(params: Dict[str, Any]) -> Dict[str, Any]:
    result = backtest_ft_1x2_core(
        start_date=params.get("start_date"),
        end_date=params.get("end_date"),
        min_edge=float(params.get("min_edge", 0.05)),
        min_ev=float(params.get("min_ev", 0.0)),
        stake=float(params.get("stake", 1.0)),
        kelly_mult=float(params.get("kelly_mult", 0.0)),
        selection_mode=params.get("selection_mode", "best_ev"),
        blend_alpha=float(params.get("blend_alpha", 1.0)),
        debug=int(params.get("debug", 0))
    )
    # Ensure serializability of dates
    if not result["bets_df"].empty:
        bets = result["bets_df"].to_dict(orient="records")
    else:
        bets = []

    return {
        "summary": result["summary"],
        "markets": result["markets"],
        "divisions": result["league_stats_df"].to_dict(orient="records") if not result["league_stats_df"].empty else [],
        "equity": result["daily_equity_df"].to_dict(orient="records") if not result["daily_equity_df"].empty else [],
        "ev_deciles": result.get("ev_deciles", []),
        "bets": bets,
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

@router.post("/sweep")
async def backtest_sweep_api(body: SweepRequest):
    params = body.dict()
    await progress_manager.broadcast({"type": "sweep_started", "payload": params})
    try:
        from functools import partial
        def progress_cb(info: Dict[str, Any]):
            try:
                anyio.from_thread.run(
                    progress_manager.broadcast,
                    {"type": "sweep_progress", "payload": info}
                )
            except Exception:
                pass

        result = await anyio.to_thread.run_sync(
            partial(run_backtest_sweep, progress_callback=progress_cb, **params)
        )
        await progress_manager.broadcast({"type": "sweep_completed", "payload": {"cells_count": len(result.get("cells", []))}})
        return result
    except Exception as exc:
        await progress_manager.broadcast({"type": "sweep_failed", "payload": {"error": str(exc)}})
        raise

@router.get("/latest-sweep")
async def get_latest_sweep_api():
    try:
        from production.backtest_sweep import load_latest_sweep
        result = await anyio.to_thread.run_sync(load_latest_sweep)
        return result
    except Exception as exc:
        return {"error": str(exc)}
