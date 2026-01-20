# api_main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anyio

from backtest_api import run_backtest_json
from ws_progress import progress_manager
from predict_live_from_history import predict_live_with_history
from train_model import run_training_process, load_features
from build_dataset import run_build_dataset_process
from features import run_build_features_process
from fastapi import BackgroundTasks
import os
import zipfile


class BacktestRequest(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    min_edge: float = 0.02
    stake: float = 1.0
    kelly_mult: float = 0.0

class TrainRequest(BaseModel):
    train_start: Optional[str] = None
    train_end: Optional[str] = None
    cutoff_date: Optional[str] = None

class LiveHistoryRequest(BaseModel):
    division: str
    match_date: str       # YYYY-MM-DD
    home_team: str
    away_team: str
    odd_home: float
    odd_draw: float
    odd_away: float
    odd_over25: Optional[float] = None
    odd_under25: Optional[float] = None
    max_odd_home: Optional[float] = None
    max_odd_draw: Optional[float] = None
    max_odd_away: Optional[float] = None
    handicap_size: Optional[float] = None
    handicap_home: Optional[float] = None
    handicap_away: Optional[float] = None

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
    
# --- Background Training ---------------------------------------------

async def background_train_task(params: dict):
    await progress_manager.broadcast({
        "type": "training_started",
        "payload": {"params": params}
    })
    try:
        # run_training_process is synchronous, so we offload it to a worker thread
        await anyio.to_thread.run_sync(
            run_training_process, 
            params.get("train_start"),
            params.get("train_end"),
            params.get("cutoff_date")
        )
        await progress_manager.broadcast({
            "type": "training_completed",
            "payload": {"status": "success"}
        })
    except Exception as exc:
        await progress_manager.broadcast({
            "type": "training_failed",
            "payload": {"error": str(exc)}
        })

@app.post("/train")
async def train_models_api(body: TrainRequest, background_tasks: BackgroundTasks):
    params = body.dict()
    background_tasks.add_task(background_train_task, params)
    return {"status": "training_dispatched", "params": params}

# --- Metadata Endpoints ----------------------------------------------

@app.get("/meta/teams")
async def get_teams():
    try:
        df = await anyio.to_thread.run_sync(load_features)
        teams = sorted(list(set(df["home_team"].unique()) | set(df["away_team"].unique())))
        return {"teams": teams}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/meta/divisions")
async def get_divisions():
    try:
        df = await anyio.to_thread.run_sync(load_features)
        divisions = sorted(df["division"].unique().tolist())
        return {"divisions": divisions}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

# --- Data Management --------------------------------------------------

def get_file_info(path: str):
    if not os.path.exists(path):
        return {"exists": False, "size": 0, "modified": None}
    stat = os.stat(path)
    return {
        "exists": True,
        "size": stat.st_size,
        "modified": stat.st_mtime
    }

@app.get("/data/status")
async def get_data_status():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Map relative to ML/data
    raw_path = os.path.join(root, "data", "raw", "Matches.csv")
    dataset_path = os.path.join(root, "data", "processed", "matches.csv")
    features_path = os.path.join(root, "data", "processed", "features.csv")
    zip_path = os.path.join(root, "data", "club-football-match-data-2000-2025.zip")
    
    return {
        "raw": get_file_info(raw_path),
        "dataset": get_file_info(dataset_path),
        "features": get_file_info(features_path),
        "zip": get_file_info(zip_path)
    }

async def background_unzip_raw():
    await progress_manager.broadcast({"type": "unzip_started", "payload": {}})
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        zip_path = os.path.join(root, "data", "club-football-match-data-2000-2025.zip")
        raw_dir = os.path.join(root, "data", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_dir)
            
        await progress_manager.broadcast({"type": "unzip_completed", "payload": {}})
    except Exception as exc:
        await progress_manager.broadcast({"type": "unzip_failed", "payload": {"error": str(exc)}})

@app.post("/data/unzip-raw")
async def unzip_raw_api(background_tasks: BackgroundTasks):
    background_tasks.add_task(background_unzip_raw)
    return {"status": "unzip_dispatched"}

async def background_build_dataset():
    await progress_manager.broadcast({"type": "dataset_build_started", "payload": {}})
    try:
        await anyio.to_thread.run_sync(run_build_dataset_process)
        await progress_manager.broadcast({"type": "dataset_build_completed", "payload": {}})
    except Exception as exc:
        await progress_manager.broadcast({"type": "dataset_build_failed", "payload": {"error": str(exc)}})

async def background_build_features():
    await progress_manager.broadcast({"type": "features_build_started", "payload": {}})
    try:
        await anyio.to_thread.run_sync(run_build_features_process)
        await progress_manager.broadcast({"type": "features_build_completed", "payload": {}})
    except Exception as exc:
        await progress_manager.broadcast({"type": "features_build_failed", "payload": {"error": str(exc)}})

@app.post("/data/build-dataset")
async def build_dataset_api(background_tasks: BackgroundTasks):
    background_tasks.add_task(background_build_dataset)
    return {"status": "dataset_build_dispatched"}

@app.post("/data/build-features")
async def build_features_api(background_tasks: BackgroundTasks):
    background_tasks.add_task(background_build_features)
    return {"status": "features_build_dispatched"}

@app.post("/predict/live-with-history")
def predict_live_with_history_api(body: LiveHistoryRequest):
    try:
        return predict_live_with_history(body.dict())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except ValueError as exc:
        # e.g. feature pipeline failed to produce row
        raise HTTPException(status_code=400, detail=str(exc))