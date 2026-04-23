"""XGBoost regression with SHAP explainability for SEO feature importance."""
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error


def engineer_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Add temporal + lag features to a date-indexed DataFrame."""
    df = df.copy()
    df["day_of_week"] = df.index.dayofweek
    df["day_of_month"] = df.index.day
    df["month"] = df.index.month
    df["week_of_year"] = df.index.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    for lag in [1, 7, 14, 28]:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

    df[f"{target_col}_rolling_7d"] = df[target_col].rolling(7, min_periods=1).mean()
    df[f"{target_col}_rolling_28d"] = df[target_col].rolling(28, min_periods=1).mean()

    return df.dropna()


def run_xgb_predict(
    features: dict[str, list[float]],
    target: str,
    dates: list[str],
) -> dict:
    """
    Train XGBoost on feature matrix and return predictions + SHAP values.

    Args:
        features: {column_name: [values...]} — must include the target column
        target: Name of the target column (e.g. "clicks")
        dates: ISO date strings aligned with the feature rows

    Returns:
        predictions, feature_importance (SHAP-based), mape
    """
    df = pd.DataFrame(features, index=pd.to_datetime(dates))
    df = df.sort_index()

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not in features")

    df = engineer_features(df, target)
    # y must be aligned with X after dropna inside engineer_features
    y = df[target]
    X = df.drop(columns=[target])
    assert len(X) == len(y), f"Feature/target length mismatch: {len(X)} vs {len(y)}"

    if len(X) < 30:
        raise ValueError("Need at least 30 rows after feature engineering")

    # Time-series split: last 20% as test
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        objective="reg:squarederror",
        random_state=42,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    preds_test = model.predict(X_test)
    mape = float(mean_absolute_percentage_error(y_test, preds_test))

    # Full predictions for output
    preds_all = model.predict(X)
    predictions = [
        {
            "date": str(d.date()),
            "predicted": round(float(p), 2),
            "actual": float(y.iloc[i]) if i < len(y) else None,
        }
        for i, (d, p) in enumerate(zip(X.index, preds_all))
    ]

    # SHAP values for feature importance
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    feature_importance = sorted(
        [
            {
                "feature": col,
                "importance": float(model.feature_importances_[i]),
                "shap_mean_abs": float(mean_abs_shap[i]),
            }
            for i, col in enumerate(X.columns)
        ],
        key=lambda x: x["shap_mean_abs"],
        reverse=True,
    )[:20]

    return {
        "predictions": predictions,
        "feature_importance": feature_importance,
        "mape": round(mape, 4),
    }
