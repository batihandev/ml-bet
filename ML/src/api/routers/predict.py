from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from production.predict import predict_live_with_history

router = APIRouter(prefix="/predict", tags=["prediction"])

class LiveHistoryRequest(BaseModel):
    division: str
    match_date: str
    home_team: str
    away_team: str
    odd_home: float
    odd_draw: float
    odd_away: float
    odd_over25: Optional[float] = None
    odd_under25: Optional[float] = None

@router.post("/live-with-history")
def predict_live_with_history_api(body: LiveHistoryRequest):
    try:
        return predict_live_with_history(body.dict())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
