"""Bayesian Structural Time Series CausalImpact analysis.

Uses the causalimpact library (Python port of Google's R CausalImpact).
Answers: "How much of my growth AFTER a given event was caused by that
event, rather than by the natural market trend (control series)?"
"""
import pandas as pd
import numpy as np
from causalimpact import CausalImpact
import logging

logging.getLogger("causalimpact").setLevel(logging.ERROR)


def run_causal_impact(
    treated: list[dict],
    control: list[dict],
    pre_period: tuple[str, str],
    post_period: tuple[str, str],
) -> dict:
    """
    Args:
        treated: Site's time-series e.g. GSC daily clicks [{"date", "value"}]
        control: Market index time-series [{"date", "value"}]
        pre_period: (start_date, end_date) before the intervention
        post_period: (start_date, end_date) after the intervention

    Returns:
        CausalImpact result dict with cumulative_effect, relative_effect,
        p_value, and plot_data for the frontend chart.
    """
    treated_df = pd.DataFrame(treated).rename(columns={"date": "ds", "value": "y"})
    treated_df["ds"] = pd.to_datetime(treated_df["ds"])

    control_df = pd.DataFrame(control).rename(columns={"date": "ds", "value": "x"})
    control_df["ds"] = pd.to_datetime(control_df["ds"])

    merged = treated_df.merge(control_df, on="ds", how="inner").set_index("ds").sort_index()

    if len(merged) < 20:
        raise ValueError("Need at least 20 overlapping data points")

    ci = CausalImpact(
        merged,
        pre_period,
        post_period,
        model_args={"nseasons": [{"period": 7}]},
    )

    summary = ci.summary()
    report = ci.summary(output="report")

    # Extract key metrics from summary table
    inferences = ci.inferences
    post_mask = (inferences.index >= post_period[0]) & (inferences.index <= post_period[1])
    post_data = inferences[post_mask]

    actual_sum = float(post_data["response"].sum()) if "response" in post_data else 0
    pred_sum = float(post_data["point_pred"].sum()) if "point_pred" in post_data else 0
    cumulative_effect = actual_sum - pred_sum
    relative_effect = (cumulative_effect / pred_sum) if pred_sum != 0 else 0

    # p-value from posterior tail probability
    p_value = float(getattr(ci, "p_value", 0.5))

    # Build plot data for the frontend chart
    plot_data = []
    for date, row in inferences.iterrows():
        is_post = post_period[0] <= str(date.date()) <= post_period[1]
        plot_data.append({
            "date": str(date.date()),
            "actual": float(row["response"]) if not is_post else float(row.get("response", row.get("point_pred", 0))),
            "predicted": float(row.get("point_pred", 0)),
            "predicted_lower": float(row.get("point_pred_lower", 0)),
            "predicted_upper": float(row.get("point_pred_upper", 0)),
        })

    return {
        "cumulative_effect": round(cumulative_effect, 2),
        "relative_effect": round(relative_effect, 4),
        "p_value": round(p_value, 6),
        "plot_data": plot_data,
        "summary": report[:2000] if isinstance(report, str) else str(summary)[:2000],
    }
