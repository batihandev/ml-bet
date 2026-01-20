from fastapi import APIRouter, BackgroundTasks
import os
import zipfile
import anyio
from pathlib import Path
from dataset import run_build_dataset_process
from features import run_build_features_process
from ..ws_progress import progress_manager

router = APIRouter(prefix="/data", tags=["data"])

# Correct root is ML/ (3 levels up from src/api/routers/data.py)
ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT_DIR / "data"

def get_file_info(path: Path):
    if not path.exists(): return {"exists": False, "size": 0, "modified": None, "name": path.name}
    stat = path.stat()
    return {"exists": True, "size": stat.st_size, "modified": stat.st_mtime, "name": path.name}

def find_zip() -> Path:
    # Try raw/ first, then data/
    paths = [
        DATA_DIR / "raw" / "club-football-match-data-2000-2025.zip",
        DATA_DIR / "club-football-match-data-2000-2025.zip",
        DATA_DIR / "raw" / "data-2000-2025.zip",
        DATA_DIR / "data-2000-2025.zip",
    ]
    for p in paths:
        if p.exists(): return p
    # Fallback to the most likely name
    return DATA_DIR / "raw" / "club-football-match-data-2000-2025.zip"

@router.get("/status")
async def get_data_status():
    zip_path = find_zip()
    return {
        "raw": get_file_info(DATA_DIR / "raw" / "Matches.csv"),
        "dataset": get_file_info(DATA_DIR / "processed" / "matches.csv"),
        "features": get_file_info(DATA_DIR / "processed" / "features.csv"),
        "zip": get_file_info(zip_path)
    }

async def background_unzip_raw():
    await progress_manager.broadcast({"type": "unzip_started", "payload": {}})
    try:
        zip_path = find_zip()
        if not zip_path.exists():
            raise FileNotFoundError(f"Zip file not found: {zip_path}")
        
        raw_dir = DATA_DIR / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref: 
            zip_ref.extractall(raw_dir)
        await progress_manager.broadcast({"type": "unzip_completed", "payload": {}})
    except Exception as exc:
        await progress_manager.broadcast({"type": "unzip_failed", "payload": {"error": str(exc)}})

@router.post("/unzip-raw")
async def unzip_raw_api(background_tasks: BackgroundTasks):
    background_tasks.add_task(background_unzip_raw)
    return {"status": "unzip_dispatched"}

@router.post("/build-dataset")
async def build_dataset_api(background_tasks: BackgroundTasks):
    background_tasks.add_task(lambda: anyio.to_thread.run_sync(run_build_dataset_process))
    return {"status": "dataset_build_dispatched"}

@router.post("/build-features")
async def build_features_api(background_tasks: BackgroundTasks):
    background_tasks.add_task(lambda: anyio.to_thread.run_sync(run_build_features_process))
    return {"status": "features_build_dispatched"}
