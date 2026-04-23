from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.deps import verify_token
from app.models.causal import run_causal_impact

router = APIRouter()


class TimeSeriesPoint(BaseModel):
    date: str
    value: float


class CausalImpactRequest(BaseModel):
    treated: list[TimeSeriesPoint]
    control: list[TimeSeriesPoint]
    pre_period: tuple[str, str]
    post_period: tuple[str, str]


@router.post("")
def causal_impact(req: CausalImpactRequest, _: dict = Depends(verify_token)):
    try:
        return run_causal_impact(
            treated=[p.model_dump() for p in req.treated],
            control=[p.model_dump() for p in req.control],
            pre_period=req.pre_period,
            post_period=req.post_period,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Causal impact failed: {e}")
