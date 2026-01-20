from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
import anyio
from training import run_training_process
from ..ws_progress import progress_manager

router = APIRouter(prefix="/train", tags=["training"])

class TrainRequest(BaseModel):
    train_start: Optional[str] = None
    train_end: Optional[str] = None
    cutoff_date: Optional[str] = None

async def background_train_task(params: dict):
    await progress_manager.broadcast({"type": "training_started", "payload": {"params": params}})
    try:
        await anyio.to_thread.run_sync(
            run_training_process, 
            params.get("train_start"),
            params.get("train_end"),
            params.get("cutoff_date")
        )
        await progress_manager.broadcast({"type": "training_completed", "payload": {"status": "success"}})
    except Exception as exc:
        await progress_manager.broadcast({"type": "training_failed", "payload": {"error": str(exc)}})

@router.post("")
async def train_models_api(body: TrainRequest, background_tasks: BackgroundTasks):
    params = body.dict()
    background_tasks.add_task(background_train_task, params)
    return {"status": "training_dispatched", "params": params}
