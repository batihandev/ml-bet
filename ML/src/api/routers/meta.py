from fastapi import APIRouter, HTTPException
import anyio
from training.data import load_features

router = APIRouter(prefix="/meta", tags=["metadata"])

@router.get("/teams")
async def get_teams():
    try:
        df = await anyio.to_thread.run_sync(load_features)
        teams = sorted(list(set(df["home_team"].unique()) | set(df["away_team"].unique())))
        return {"teams": teams}
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc))

@router.get("/divisions")
async def get_divisions():
    try:
        df = await anyio.to_thread.run_sync(load_features)
        divisions = sorted(df["division"].unique().tolist())
        return {"divisions": divisions}
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc))
