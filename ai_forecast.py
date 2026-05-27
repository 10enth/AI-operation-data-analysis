"""AI 销量预测引擎 — 多模型自动训练 + 最优选择 + 预测"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")


def _prepare_prophet_data(df, target, date_col):
    """将 DataFrame 转为 Prophet 需要的 ds/y 格式"""
    df = df[[date_col, target]].copy()
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"])
    return df.dropna()


def _build_lgb_features(df, target, date_col, lags=4, window=4):
    """为 LightGBM 构造时间序列监督特征"""
    data = df[[date_col, target]].copy()
    data["ds"] = pd.to_datetime(data[date_col])
    data = data.sort_values("ds")

    for lag in range(1, lags + 1):
        data[f"lag_{lag}"] = data[target].shift(lag)
    data["rolling_mean"] = data[target].shift(1).rolling(window).mean()
    data["month"] = data["ds"].dt.month
    data["quarter"] = data["ds"].dt.quarter

    return data.dropna()


def _evaluate(y_true, y_pred):
    """计算 RMSE 和 MAE"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {"rmse": round(rmse, 2), "mae": round(mae, 2)}


def compare_models(df, target, date_col):
    """
    对比多个模型的预测性能。

    参数:
        df: DataFrame, 时间序列数据
        target: str, 目标列名
        date_col: str, 日期列名

    返回:
        [{model_name, rmse, mae}], 按 RMSE 升序
    """
    from statsmodels.tsa.arima.model import ARIMA
    from prophet import Prophet
    from lightgbm import LGBMRegressor

    series = df[target].values.astype(float)
    dates = pd.to_datetime(df[date_col])
    n = len(series)
    split = int(n * 0.8)
    train_series = series[:split]
    test_series = series[split:]
    results = []

    # --- ARIMA ---
    try:
        model = ARIMA(train_series, order=(2, 1, 2))
        fitted = model.fit()
        pred = fitted.forecast(steps=len(test_series))
        metrics = _evaluate(test_series, pred)
        metrics["model_name"] = "ARIMA"
        results.append(metrics)
    except Exception:
        results.append({"model_name": "ARIMA", "rmse": float("inf"), "mae": float("inf")})

    # --- Prophet ---
    try:
        prop_data = _prepare_prophet_data(df.iloc[:split], target, date_col)
        m = Prophet(yearly_seasonality=False, weekly_seasonality=True,
                     daily_seasonality=False, seasonality_mode="additive")
        m.fit(prop_data)
        future = m.make_future_dataframe(periods=len(test_series), freq="W")
        forecast = m.predict(future)
        pred = forecast["yhat"].values[-len(test_series):]
        metrics = _evaluate(test_series, pred)
        metrics["model_name"] = "Prophet"
        results.append(metrics)
    except Exception:
        results.append({"model_name": "Prophet", "rmse": float("inf"), "mae": float("inf")})

    # --- LightGBM ---
    try:
        featurized = _build_lgb_features(df, target, date_col)
        feature_cols = [c for c in featurized.columns
                        if c not in [target, date_col, "ds"]
                        and not featurized[c].dtype == "object"]
        X = featurized[feature_cols].values
        y = featurized[target].values
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        lgb = LGBMRegressor(n_estimators=100, max_depth=6, random_state=42, verbose=-1)
        lgb.fit(X_train, y_train)
        pred = lgb.predict(X_test)
        metrics = _evaluate(y_test, pred)
        metrics["model_name"] = "LightGBM"
        results.append(metrics)
    except Exception:
        results.append({"model_name": "LightGBM", "rmse": float("inf"), "mae": float("inf")})

    results.sort(key=lambda x: x["rmse"])
    return results


def train_and_forecast(df, target, date_col, periods=8):
    """
    训练多模型、选最优、预测未来。

    参数:
        df: DataFrame, 时间序列数据
        target: str, 目标列名
        date_col: str, 日期列名
        periods: int, 预测周期数

    返回:
        {
            "forecast": [values],
            "conf_low": [values],
            "conf_high": [values],
            "best_model": str,
            "metrics": {rmse, mae},
            "model_comparison": [{model_name, rmse, mae}, ...],
            "future_dates": [dates],
        }
    """
    from statsmodels.tsa.arima.model import ARIMA
    from prophet import Prophet
    from lightgbm import LGBMRegressor

    series = df[target].values.astype(float)
    dates = pd.to_datetime(df[date_col])
    n = len(series)

    # 对比模型
    comparison = compare_models(df, target, date_col)
    best_name = comparison[0]["model_name"]

    best_metrics = {"rmse": comparison[0]["rmse"], "mae": comparison[0]["mae"]}
    forecast = None
    conf_low = None
    conf_high = None

    # --- 用最优模型在全量数据上训练并预测 ---
    if best_name == "ARIMA":
        model = ARIMA(series, order=(2, 1, 2))
        fitted = model.fit()
        fc = fitted.get_forecast(steps=periods)
        forecast = fc.predicted_mean.round(2).tolist()
        ci = fc.conf_int()
        conf_low = ci[:, 0].round(2).tolist()
        conf_high = ci[:, 1].round(2).tolist()
        best_metrics["aic"] = round(fitted.aic, 2) if hasattr(fitted, "aic") else None

    elif best_name == "Prophet":
        prop_data = _prepare_prophet_data(df, target, date_col)
        m = Prophet(yearly_seasonality=False, weekly_seasonality=True,
                     daily_seasonality=False, seasonality_mode="additive")
        m.fit(prop_data)
        future = m.make_future_dataframe(periods=periods, freq="W")
        fc = m.predict(future)
        forecast = fc["yhat"].values[-periods:].round(2).tolist()
        conf_low = fc["yhat_lower"].values[-periods:].round(2).tolist()
        conf_high = fc["yhat_upper"].values[-periods:].round(2).tolist()

    else:  # LightGBM
        featurized = _build_lgb_features(df, target, date_col)
        feature_cols = [c for c in featurized.columns
                        if c not in [target, date_col, "ds"]
                        and not featurized[c].dtype == "object"]
        X = featurized[feature_cols].values
        y = featurized[target].values
        lgb = LGBMRegressor(n_estimators=100, max_depth=6, random_state=42, verbose=-1)
        lgb.fit(X, y)

        # 递归预测
        last_row = featurized.iloc[-1]
        predictions = []
        current = last_row[feature_cols].values.copy()
        for _ in range(periods):
            pred = lgb.predict(current.reshape(1, -1))[0]
            predictions.append(pred)
            # 滚动特征
            for j in range(len(feature_cols) - 2, -1, -1):
                if j > 0:
                    current[j] = current[j - 1]
            current[0] = pred

        forecast = [round(p, 2) for p in predictions]
        # 置信区间: 用 RMSE 估算
        rmse_val = comparison[0]["rmse"]
        conf_low = [round(p - 1.96 * rmse_val, 2) for p in predictions]
        conf_high = [round(p + 1.96 * rmse_val, 2) for p in predictions]

    # 生成未来日期
    freq = pd.infer_freq(dates)
    if freq is None:
        freq = "W-MON"
    future_dates = pd.date_range(start=dates.iloc[-1] + pd.Timedelta(weeks=1),
                                  periods=periods, freq=freq)
    future_dates_str = [d.strftime("%Y-%m-%d") for d in future_dates]

    return {
        "forecast": forecast,
        "conf_low": conf_low,
        "conf_high": conf_high,
        "best_model": best_name,
        "metrics": best_metrics,
        "model_comparison": comparison,
        "future_dates": future_dates_str,
    }


def explain_forecast(forecast_result):
    """
    用 LLM 生成预测结果的中文解读。

    参数:
        forecast_result: train_and_forecast() 的返回值

    返回:
        str, 中文预测解读
    """
    from ai_insight import generate_insight

    context = {
        "best_model": forecast_result["best_model"],
        "rmse": forecast_result["metrics"]["rmse"],
        "forecast": forecast_result["forecast"],
        "future_dates": forecast_result["future_dates"],
        "model_ranking": [m["model_name"] for m in forecast_result["model_comparison"]],
    }
    result = generate_insight(context, "forecast")
    return result
