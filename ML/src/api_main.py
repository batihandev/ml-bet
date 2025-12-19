# api_main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anyio

from backtest_api import run_backtest_json
from ws_progress import progress_manager
from predict_live_from_history import predict_live_with_history


class BacktestRequest(BaseModel):
    start_date: str | None = None
    end_date: str | None = None
    min_edge: float = 0.02
    stake: float = 1.0
    kelly_mult: float = 0.0

class LiveHistoryRequest(BaseModel):
    division: str
    match_date: str       # YYYY-MM-DD
    home_team: str
    away_team: str
    odd_home: float
    odd_draw: float
    odd_away: float
    odd_over25: float | None = None
    odd_under25: float | None = None
    max_odd_home: float | None = None
    max_odd_draw: float | None = None
    max_odd_away: float | None = None
    handicap_size: float | None = None
    handicap_home: float | None = None
    handicap_away: float | None = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3005",
        "http://127.0.0.1:3005",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- WebSocket: generic progress channel ------------------------------


@app.websocket("/ws/progress")
async def websocket_progress(websocket: WebSocket):
    """
    Simple broadcast channel: server pushes job events as JSON messages.

    Example payloads:
      { "type": "backtest_started", "payload": { ... } }
      { "type": "backtest_completed", "payload": { "summary": {...} } }
      { "type": "backtest_failed", "payload": { "error": "..." } }
    """
    await progress_manager.connect(websocket)
    try:
        # Keep the connection open; we ignore any messages from client for now.
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await progress_manager.disconnect(websocket)


# --- HTTP backtest endpoint ------------------------------------------


@app.post("/backtest/ft-1x2")
async def backtest_ft_1x2_api(body: BacktestRequest):
    params = body.dict()

    # Tell all listeners a backtest started
    await progress_manager.broadcast(
        {
            "type": "backtest_started",
            "payload": {
                "job": "ft_1x2",
                "params": params,
            },
        }
    )

    try:
        # run_backtest_json is synchronous, so we offload it to a worker thread
        result = await anyio.to_thread.run_sync(run_backtest_json, params)

        # Notify completion with a short summary
        await progress_manager.broadcast(
            {
                "type": "backtest_completed",
                "payload": {
                    "job": "ft_1x2",
                    "summary": result.get("summary", {}),
                },
            }
        )
        return result
    except Exception as exc:
        # Notify failure
        await progress_manager.broadcast(
            {
                "type": "backtest_failed",
                "payload": {
                    "job": "ft_1x2",
                    "error": str(exc),
                },
            }
        )
        raise

@app.post("/predict/live-with-history")
def predict_live_with_history_api(body: LiveHistoryRequest):
    try:
        return predict_live_with_history(body.dict())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except ValueError as exc:
        # e.g. feature pipeline failed to produce row
        raise HTTPException(status_code=400, detail=str(exc))