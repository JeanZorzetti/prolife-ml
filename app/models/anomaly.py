"""Prophet-based anomaly detection using uncertainty intervals."""
import pandas as pd
import numpy as np
from prophet import Prophet
import logging

logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)


def detect_anomalies(series: list[dict], sensitivity: float = 2.0) -> dict:
    """
    Fit a Prophet model and flag points outside the uncertainty interval as anomalies.

    Args:
        series: List of {"date": "YYYY-MM-DD", "value": float}
        sensitivity: Z-score multiplier for the anomaly threshold (default 2.0 = ~95%)

    Returns:
        dict with list of AnomalyPoint
    """
    df = pd.DataFrame(series).rename(columns={"date": "ds", "value": "y"})
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").dropna()

    if len(df) < 14:
        raise ValueError("Need at least 14 data points for anomaly detection")

    m = Prophet(
        interval_width=0.95,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        uncertainty_samples=500,
    )
    m.fit(df)
    forecast = m.predict(df)

    merged = df.merge(
        forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]],
        on="ds",
    )

    # Compute residuals and z-scores
    merged["residual"] = merged["y"] - merged["yhat"]
    std = merged["residual"].std()
    merged["z_score"] = merged["residual"] / std if std > 0 else 0.0

    anomalies = []
    for _, row in merged[merged["z_score"].abs() > sensitivity].iterrows():
        z = float(row["z_score"])
        if abs(z) > 3.5:
            sev = "high"
        elif abs(z) > 2.5:
            sev = "medium"
        else:
            sev = "low"

        anomalies.append({
            "date": str(row["ds"].date()),
            "value": float(row["y"]),
            "expected": float(row["yhat"]),
            "z_score": round(z, 2),
            "severity": sev,
        })

    return {"anomalies": sorted(anomalies, key=lambda a: a["date"])}
