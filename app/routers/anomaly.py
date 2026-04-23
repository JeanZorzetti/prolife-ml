from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.deps import verify_token
from app.models.anomaly import detect_anomalies

router = APIRouter()


class TimeSeriesPoint(BaseModel):
    date: str
    value: float


class AnomalyRequest(BaseModel):
    series: list[TimeSeriesPoint]
    sensitivity: float = 2.0


@router.post("")
@router.post("/")
def anomaly(req: AnomalyRequest, _: dict = Depends(verify_token)):
    try:
        return detect_anomalies(
            series=[p.model_dump() for p in req.series],
            sensitivity=req.sensitivity,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {e}")
