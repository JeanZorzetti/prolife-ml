"""XGBoost regression with SHAP explainability for SEO feature importance."""
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.metrics import mean_absolute_percentage_error


def engineer_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Add temporal + lag features. Uses smaller lags when data is short."""
    df = df.copy()
    df["day_of_week"] = df.index.dayofweek
    df["day_of_month"] = df.index.day
    df["month"] = df.index.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    n = len(df)
    # Only add lags that won't eat more than 50% of the data
    lags = [l for l in [1, 7, 14, 28] if l < n * 0.5]
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

    roll = min(7, n // 3)
    df[f"{target_col}_rolling_{roll}d"] = df[target_col].rolling(roll, min_periods=1).mean()

    return df.dropna()


def run_xgb_predict(
    features: dict[str, list[float]],
    target: str,
    dates: list[str],
) -> dict:
    df = pd.DataFrame(features, index=pd.to_datetime(dates))
    df = df.sort_index()

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not in features")

    df = engineer_features(df, target)

    y = df[target]
    X = df.drop(columns=[target])

    if len(X) < 10:
        raise ValueError(f"Need at least 10 rows after feature engineering (got {len(X)})")

    # Time-series split: last 20% as test, min 5 rows for test
    split = max(int(len(X) * 0.8), len(X) - 5)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds_test = model.predict(X_test)
    mape = float(mean_absolute_percentage_error(y_test, preds_test))

    preds_all = model.predict(X)
    predictions = [
        {
            "date": str(d.date()),
            "predicted": round(float(p), 2),
            "actual": float(y.iloc[i]),
        }
        for i, (d, p) in enumerate(zip(X.index, preds_all))
    ]

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
