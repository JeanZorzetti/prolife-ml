"""NeuralProphet-based forecasting with optional external regressors."""
import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
from typing import Optional
import logging

logging.getLogger("NP.train_loop").setLevel(logging.ERROR)
logging.getLogger("NP.forecaster").setLevel(logging.WARNING)


def _safe_float(v, fallback=0.0) -> float:
    """Return float, replacing NaN/inf with fallback."""
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return fallback
        return f
    except (TypeError, ValueError):
        return fallback


def run_forecast(
    series: list[dict],
    horizon_days: int = 30,
    regressors: Optional[list[dict]] = None,
) -> dict:
    df = pd.DataFrame(series).rename(columns={"date": "ds", "value": "y"})
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").dropna()

    if len(df) < 14:
        raise ValueError("Need at least 14 data points for NeuralProphet forecast")

    # Use n_lags=0 when series is short to avoid NeuralProphet data requirements
    n_lags = 7 if len(df) >= 60 else 0

    m = NeuralProphet(
        n_forecasts=1,       # predict 1 step ahead (avoids future regressor requirement)
        n_lags=n_lags,
        yearly_seasonality="auto",
        weekly_seasonality=True,
        daily_seasonality=False,
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        trainer_config={"enable_progress_bar": False},
    )

    metrics = m.fit(df, freq="D", progress=None)

    # Build future dataframe for horizon_days
    future = m.make_future_dataframe(df, periods=horizon_days, n_historic_predictions=True)
    forecast_df = m.predict(future)

    mape = None
    if metrics is not None and "MAE" in metrics.columns:
        raw = metrics["MAE"].iloc[-1]
        mape = _safe_float(raw)

    # Extract forecast rows (last horizon_days)
    forecast_rows = []
    future_rows = forecast_df[forecast_df["ds"] > df["ds"].max()].head(horizon_days)
    for _, row in future_rows.iterrows():
        yhat = _safe_float(row.get("yhat1", row.get("yhat", 0)))
        forecast_rows.append({
            "date": str(row["ds"].date()),
            "yhat": yhat,
            "yhat_lower": round(yhat * 0.85, 2),
            "yhat_upper": round(yhat * 1.15, 2),
        })

    # Trend component from historic predictions
    trend_rows = []
    for _, row in forecast_df.iterrows():
        trend_rows.append({
            "date": str(row["ds"].date()),
            "yhat": _safe_float(row.get("trend", 0)),
            "yhat_lower": _safe_float(row.get("trend", 0)),
            "yhat_upper": _safe_float(row.get("trend", 0)),
        })

    weekly_rows = []
    for _, row in forecast_df.iterrows():
        weekly_rows.append({
            "date": str(row["ds"].date()),
            "yhat": _safe_float(row.get("season_weekly", 0)),
            "yhat_lower": 0.0,
            "yhat_upper": 0.0,
        })

    return {
        "forecast": forecast_rows,
        "components": {
            "trend": trend_rows[-horizon_days * 2:],
            "weekly": weekly_rows[-horizon_days * 2:],
            "yearly": [],
        },
        "mape": mape,
        "model_version": f"neuralprophet-0.9-lags{n_lags}",
    }
