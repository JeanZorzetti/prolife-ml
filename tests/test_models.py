"""Smoke tests for all 4 ML models using synthetic data."""
import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta


def make_series(n=60, seed=42):
    rng = np.random.default_rng(seed)
    base = 100
    dates = [(date(2025, 1, 1) + timedelta(days=i)).isoformat() for i in range(n)]
    values = [base + i * 0.5 + rng.normal(0, 5) for i in range(n)]
    return [{"date": d, "value": max(0, v)} for d, v in zip(dates, values)]


def make_control(series):
    return [{"date": p["date"], "value": p["value"] * 0.8 + np.random.normal(0, 2)} for p in series]


# ─── NeuralProphet forecast ────────────────────────────────────────────────
def test_forecast_smoke():
    from app.models.forecast import run_forecast
    series = make_series(60)
    result = run_forecast(series, horizon_days=14)
    assert len(result["forecast"]) == 14
    assert "mape" in result
    assert result["forecast"][0]["yhat"] > 0


def test_forecast_too_short():
    from app.models.forecast import run_forecast
    with pytest.raises(ValueError, match="at least 14"):
        run_forecast(make_series(10), horizon_days=7)


# ─── Prophet anomaly detection ────────────────────────────────────────────
def test_anomaly_smoke():
    from app.models.anomaly import detect_anomalies
    series = make_series(60)
    # Inject obvious spike
    series[30] = {"date": series[30]["date"], "value": 500.0}
    result = detect_anomalies(series, sensitivity=2.0)
    assert "anomalies" in result
    # Should detect at least the spike
    assert len(result["anomalies"]) >= 1
    dates = [a["date"] for a in result["anomalies"]]
    assert series[30]["date"] in dates


def test_anomaly_too_short():
    from app.models.anomaly import detect_anomalies
    with pytest.raises(ValueError, match="at least 14"):
        detect_anomalies(make_series(10))


# ─── CausalImpact ─────────────────────────────────────────────────────────
def test_causal_impact_smoke():
    from app.models.causal import run_causal_impact
    series = make_series(60)
    control = make_control(series)
    # Inject treatment effect after day 40
    for i in range(40, 60):
        series[i]["value"] += 30
    result = run_causal_impact(
        treated=series,
        control=control,
        pre_period=(series[0]["date"], series[39]["date"]),
        post_period=(series[40]["date"], series[59]["date"]),
    )
    assert "cumulative_effect" in result
    assert "relative_effect" in result
    assert "p_value" in result
    assert len(result["plot_data"]) > 0


def test_causal_impact_too_short():
    from app.models.causal import run_causal_impact
    with pytest.raises(ValueError, match="at least 20"):
        series = make_series(10)
        run_causal_impact(
            treated=series, control=make_control(series),
            pre_period=(series[0]["date"], series[5]["date"]),
            post_period=(series[6]["date"], series[9]["date"]),
        )


# ─── XGBoost + SHAP ───────────────────────────────────────────────────────
def test_xgb_smoke():
    from app.models.xgb import run_xgb_predict
    series = make_series(60)
    dates = [p["date"] for p in series]
    clicks = [p["value"] for p in series]
    market_index = [v * 0.9 + np.random.normal(0, 1) for v in clicks]

    result = run_xgb_predict(
        features={"clicks": clicks, "market_index": market_index},
        target="clicks",
        dates=dates,
    )
    assert len(result["predictions"]) == len(series)
    assert len(result["feature_importance"]) > 0
    assert result["mape"] >= 0


def test_xgb_missing_target():
    from app.models.xgb import run_xgb_predict
    with pytest.raises(ValueError, match="Target column"):
        run_xgb_predict(
            features={"market_index": [1.0] * 40},
            target="clicks",
            dates=[(date(2025, 1, 1) + timedelta(days=i)).isoformat() for i in range(40)],
        )
