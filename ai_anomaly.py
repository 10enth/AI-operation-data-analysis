"""AI 异常检测引擎 — 多维度统计检测 + 分级预警"""
import numpy as np
import pandas as pd


def _iqr_detect(series, k=1.5):
    """IQR 异常值检测，返回异常索引和分数"""
    q1, q3 = np.percentile(series, 25), np.percentile(series, 75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    scores = []
    for v in series:
        if v < lower:
            score = min(100, (lower - v) / (iqr + 1e-8) * 50)
        elif v > upper:
            score = min(100, (v - upper) / (iqr + 1e-8) * 50)
        else:
            score = 0
        scores.append(round(score, 1))
    return scores, lower, upper


def _sigma_detect(series, window=4):
    """基于滚动均值和标准差的异常检测"""
    s = pd.Series(series)
    rolling_mean = s.rolling(window, min_periods=2).mean()
    rolling_std = s.rolling(window, min_periods=2).std().fillna(0)
    scores = []
    for i, (v, m, s) in enumerate(zip(series, rolling_mean, rolling_std)):
        if s == 0:
            scores.append(0)
        else:
            z = abs(v - m) / s
            scores.append(round(min(100, z * 25), 1))
    threshold_1s = rolling_mean + rolling_std
    threshold_2s = rolling_mean + 2 * rolling_std
    return scores, threshold_1s, threshold_2s


def _trend_detect(series, min_consecutive=3):
    """检测连续上升/下降趋势"""
    if len(series) < min_consecutive:
        return False, 0, ""

    recent = series.iloc[-min_consecutive:]
    if len(recent) < min_consecutive:
        return False, 0, ""

    increasing = all(recent.iloc[i] > recent.iloc[i - 1] for i in range(1, len(recent)))
    decreasing = all(recent.iloc[i] < recent.iloc[i - 1] for i in range(1, len(recent)))

    if increasing:
        change = (recent.iloc[-1] - recent.iloc[0]) / (abs(recent.iloc[0]) + 1e-8) * 100
        return True, change, "上升"
    elif decreasing:
        change = (recent.iloc[0] - recent.iloc[-1]) / (abs(recent.iloc[0]) + 1e-8) * 100
        return True, change, "下降"
    return False, 0, ""


def score_anomaly(value, baseline, method="iqr"):
    """
    对单个指标值进行异常评分。

    返回:
        {score: float(0-100), level: str}
    """
    if method == "deviation":
        # 百分比偏差
        if baseline == 0:
            pct = 0
        else:
            pct = abs(value - baseline) / abs(baseline) * 100

        score = min(100, pct * 2)
        if pct < 15:
            level = "normal"
        elif pct < 25:
            level = "注意"
        elif pct < 40:
            level = "警告"
        else:
            level = "严重"
    else:
        score = min(100, abs(value - baseline) / (abs(baseline) + 1e-8) * 100)
        level = "注意" if score > 20 else ("警告" if score > 50 else ("严重" if score > 80 else "normal"))

    return {"score": round(score, 1), "level": level}


LEVEL_EMOJI = {"normal": "✅", "注意": "🟡", "警告": "🟠", "严重": "🔴"}


def detect_anomalies(weekly_df, forecast_result=None):
    """
    对运营指标进行多维度异常检测。

    参数:
        weekly_df: DataFrame, 按周聚合的数据
                   必须包含列: date, revenue, orders, return_rate, profit
        forecast_result: dict, train_and_forecast() 的返回值 (可选)

    返回:
        list, [{metric, date, value, expected_range, level, score, description}]
    """
    anomalies = []
    df = weekly_df.sort_values("date").reset_index(drop=True)

    # 1. 营收 — IQR 检测
    if "revenue" in df.columns:
        scores, lower, upper = _iqr_detect(df["revenue"].values)
        for i, (s, v) in enumerate(zip(scores, df["revenue"].values)):
            if s > 30:
                anomalies.append({
                    "metric": "营收",
                    "date": str(df["date"].iloc[i]),
                    "value": round(v, 2),
                    "expected_range": f"${lower:,.0f} - ${upper:,.0f}",
                    "level": "警告" if s > 60 else "注意",
                    "score": s,
                    "description": f"营收 ${v:,.0f} 超出正常范围",
                })

    # 2. 营收 vs 预测偏差（如果有预测）
    if forecast_result and "revenue" in df.columns:
        forecast_vals = forecast_result.get("forecast", [])
        if forecast_vals:
            latest = df["revenue"].values[-1]
            predicted = forecast_vals[0]  # 最近一周的预测值
            result = score_anomaly(latest, predicted, "deviation")
            if result["level"] != "normal":
                pct = abs(latest - predicted) / predicted * 100
                direction = "低于" if latest < predicted else "高于"
                anomalies.append({
                    "metric": "营收",
                    "date": str(df["date"].iloc[-1]),
                    "value": round(latest, 2),
                    "expected_range": f"预测: ${predicted:,.0f}",
                    "level": result["level"],
                    "score": result["score"],
                    "description": f"本周营收 ${latest:,.0f}，{direction}预测值 {pct:.1f}%",
                })

    # 3. 退货率 — 趋势 + Sigma 检测
    if "return_rate" in df.columns:
        trend, change_pct, direction = _trend_detect(df["return_rate"])
        if trend and direction == "上升" and change_pct > 10:
            latest = df["return_rate"].values[-1]
            mean_val = df["return_rate"].mean()
            level = "警告" if change_pct > 30 else "注意"
            anomalies.append({
                "metric": "退货率",
                "date": str(df["date"].iloc[-1]),
                "value": round(latest * 100, 2),
                "expected_range": f"均值: {mean_val*100:.1f}%",
                "level": level,
                "score": min(100, change_pct * 2),
                "description": f"退货率连续上升 {change_pct:.1f}%，当前 {latest*100:.2f}%",
            })

        # Sigma 检测
        scores, th1, th2 = _sigma_detect(df["return_rate"].values)
        for i, (s, v) in enumerate(zip(scores, df["return_rate"].values)):
            if s > 60:
                anomalies.append({
                    "metric": "退货率",
                    "date": str(df["date"].iloc[i]),
                    "value": round(v * 100, 2),
                    "expected_range": f"正常范围: <{th1.iloc[i]*100:.1f}%",
                    "level": "严重" if s > 80 else "警告",
                    "score": s,
                    "description": f"退货率 {v*100:.2f}% 突破 {2 if s > 80 else 1}σ 界限",
                })

    # 4. 订单量 — IQR 检测
    if "orders" in df.columns:
        scores, lower, upper = _iqr_detect(df["orders"].values)
        for i, (s, v) in enumerate(zip(scores, df["orders"].values)):
            if s > 40:
                anomalies.append({
                    "metric": "订单量",
                    "date": str(df["date"].iloc[i]),
                    "value": int(v),
                    "expected_range": f"{int(lower)} - {int(upper)}",
                    "level": "警告" if s > 60 else "注意",
                    "score": s,
                    "description": f"订单量 {int(v)} 偏离正常范围",
                })

    # 5. 利润率 — 移动平均偏离
    if "profit_margin" in df.columns:
        ma = df["profit_margin"].rolling(4, min_periods=2).mean()
        for i in range(len(df)):
            if pd.isna(ma.iloc[i]):
                continue
            v = df["profit_margin"].iloc[i]
            m = ma.iloc[i]
            if m > 0 and v < m * 0.9:
                pct = (m - v) / m * 100
                anomalies.append({
                    "metric": "利润率",
                    "date": str(df["date"].iloc[i]),
                    "value": round(v * 100, 2),
                    "expected_range": f"4周均线: {m*100:.1f}%",
                    "level": "注意",
                    "score": min(100, pct * 3),
                    "description": f"利润率 {v*100:.1f}% 低于4周均线 {m*100:.1f}%",
                })

    # 按严重程度排序: score 降序
    anomalies.sort(key=lambda x: x["score"], reverse=True)

    # 去重：同一 metric + date 只保留 score 最高的
    seen = set()
    unique = []
    for a in anomalies:
        key = (a["metric"], a["date"])
        if key not in seen:
            seen.add(key)
            unique.append(a)

    return unique
