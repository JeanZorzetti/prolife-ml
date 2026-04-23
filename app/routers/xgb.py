from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.deps import verify_token
from app.models.xgb import run_xgb_predict

router = APIRouter()


class XgbRequest(BaseModel):
    features: dict[str, list[float]]
    target: str
    dates: list[str]


@router.post("")
@router.post("/")
def xgb_predict(req: XgbRequest, _: dict = Depends(verify_token)):
    try:
        return run_xgb_predict(
            features=req.features,
            target=req.target,
            dates=req.dates,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"XGBoost prediction failed: {e}")
