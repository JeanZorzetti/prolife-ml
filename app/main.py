from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.deps import verify_token
from app.routers import forecast, anomaly, causal, xgb

app = FastAPI(
    title="prolife-ml",
    description="ML time-series microservice for ProLife SEO analytics",
    version="1.0.0",
    docs_url="/docs",  # Disable in prod by setting to None
)

# Only allow calls from the Next.js service (same EasyPanel network)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://prolife-site:3000", "http://localhost:3000"],
    allow_methods=["POST", "GET"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(forecast.router, prefix="/forecast", tags=["forecast"])
app.include_router(anomaly.router, prefix="/anomaly", tags=["anomaly"])
app.include_router(causal.router, prefix="/causal-impact", tags=["causal"])
app.include_router(xgb.router, prefix="/xgb-predict", tags=["xgboost"])


@app.get("/health")
def health():
    return {"ok": True, "service": "prolife-ml"}
