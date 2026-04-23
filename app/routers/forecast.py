from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.deps import verify_token
from app.models.forecast import run_forecast

router = APIRouter()


class TimeSeriesPoint(BaseModel):
    date: str
    value: float


class Regressor(BaseModel):
    name: str
    series: list[TimeSeriesPoint]


class ForecastRequest(BaseModel):
    series: list[TimeSeriesPoint]
    horizon_days: int = 30
    regressors: Optional[list[Regressor]] = None


@router.post("")
def forecast(req: ForecastRequest, _: dict = Depends(verify_token)):
    try:
        result = run_forecast(
            series=[p.model_dump() for p in req.series],
            horizon_days=req.horizon_days,
            regressors=[{"name": r.name, "series": [p.model_dump() for p in r.series]} for r in (req.regressors or [])],
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast failed: {e}")
