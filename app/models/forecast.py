"""NeuralProphet-based forecasting with optional external regressors."""
import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
from typing import Optional


def run_forecast(
    series: list[dict],
    horizon_days: int = 30,
    regressors: Optional[list[dict]] = None,
) -> dict:
    """
    Train NeuralProphet on a time-series and return forecast + components.

    Args:
        series: List of {"date": "YYYY-MM-DD", "value": float}
        horizon_days: Number of future days to forecast
        regressors: Optional list of {"name": str, "series": [{"date", "value"}]}
            Used as exogenous regressors (e.g., market index).

    Returns:
        dict with forecast, components, mape, model_version
    """
    df = pd.DataFrame(series).rename(columns={"date": "ds", "value": "y"})
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").dropna()

    if len(df) < 14:
        raise ValueError("Need at least 14 data points for NeuralProphet forecast")

    # Add regressors to training DataFrame
    regressor_names = []
    if regressors:
        for reg in regressors:
            reg_df = pd.DataFrame(reg["series"]).rename(
                columns={"date": "ds", "value": reg["name"]}
            )
            reg_df["ds"] = pd.to_datetime(reg_df["ds"])
            df = df.merge(reg_df[["ds", reg["name"]]], on="ds", how="left")
            df[reg["name"]] = df[reg["name"]].ffill().bfill()
            regressor_names.append(reg["name"])

    m = NeuralProphet(
        n_forecasts=horizon_days,
        n_lags=7,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        epochs=200,
        batch_size=32,
        learning_rate=0.001,
        trainer_config={"enable_progress_bar": False},
    )

    for name in regressor_names:
        m.add_future_regressor(name)

    # Suppress NeuralProphet's verbose output
    import logging
    logging.getLogger("NP.train_loop").setLevel(logging.ERROR)

    metrics = m.fit(df, freq="D", progress=None)

    # Build future dataframe (fill regressors with last known value)
    future = m.make_future_dataframe(df, n_historic_predictions=True)
    if regressor_names:
        for name in regressor_names:
            last_val = df[name].iloc[-1]
            future[name] = future[name].fillna(last_val)

    forecast_df = m.predict(future)

    # Extract MAE/MAPE from training metrics
    mape = float(metrics["MAE"].iloc[-1]) if "MAE" in metrics.columns else None

    # Build output
    forecast_rows = []
    for _, row in forecast_df.tail(horizon_days).iterrows():
        forecast_rows.append({
            "date": str(row["ds"].date()),
            "yhat": float(row.get("yhat1", row.get("yhat", 0))),
            "yhat_lower": float(row.get("yhat1 5.0%", row.get("yhat1", 0)) * 0.9),
            "yhat_upper": float(row.get("yhat1 95.0%", row.get("yhat1", 0)) * 1.1),
        })

    # Trend component
    trend_rows = []
    for _, row in forecast_df.iterrows():
        trend_rows.append({
            "date": str(row["ds"].date()),
            "yhat": float(row.get("trend", 0)),
            "yhat_lower": float(row.get("trend", 0)),
            "yhat_upper": float(row.get("trend", 0)),
        })

    # Weekly seasonality component
    weekly_rows = []
    for _, row in forecast_df.iterrows():
        weekly_rows.append({
            "date": str(row["ds"].date()),
            "yhat": float(row.get("season_weekly", 0)),
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
        "model_version": "neuralprophet-0.9",
    }
