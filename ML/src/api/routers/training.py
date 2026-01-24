from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional
import anyio
import json
from pathlib import Path
from production.train import train_production_model as run_training_process
from production.train import MODELS_DIR
from ..ws_progress import progress_manager

router = APIRouter(prefix="/train", tags=["training"])

class TrainRequest(BaseModel):
    train_start: Optional[str] = None
    train_end: Optional[str] = None
    cutoff_date: Optional[str] = None
    n_estimators: int = 300
    max_depth: int = 8
    min_samples_leaf: int = 50

async def background_train_task(params: dict):
    await progress_manager.broadcast({"type": "training_started", "payload": {"params": params}})
    try:
        await anyio.to_thread.run_sync(
            run_training_process, 
            params.get("train_start"),
            params.get("train_end"),
            params.get("cutoff_date"),
            params.get("n_estimators", 300),
            params.get("max_depth", 8),
            params.get("min_samples_leaf", 50)
        )
        await progress_manager.broadcast({"type": "training_completed", "payload": {"status": "success"}})
    except Exception as exc:
        await progress_manager.broadcast({"type": "training_failed", "payload": {"error": str(exc)}})

@router.post("")
async def train_models_api(body: TrainRequest, background_tasks: BackgroundTasks):
    params = body.dict()
    background_tasks.add_task(background_train_task, params)
    return {"status": "training_dispatched", "params": params}

@router.get("/models")
async def list_models():
    if not MODELS_DIR.exists():
        return {"models": []}
    
    try:
        def get_model_names():
            # Find all files ending in _meta.json and strip the suffix
            return sorted([f.name.replace("_meta.json", "") for f in MODELS_DIR.glob("*_meta.json")])
        
        models = await anyio.to_thread.run_sync(get_model_names)
        return {"models": models}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@router.get("/feature-importance/{model_name}")
async def get_feature_importance(model_name: str):
    meta_path = MODELS_DIR / f"{model_name}_meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Model metadata not found. Please train the model first.")
    
    try:
        def read_meta():
            with open(meta_path, "r") as f:
                return json.load(f)
        
        meta = await anyio.to_thread.run_sync(read_meta)
        return {
            "model_name": model_name,
            **meta
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
