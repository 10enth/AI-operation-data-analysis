"""
AI 运营数据分析平台
Streamlit + Plotly + ML 模型 + LLM 洞察
上传 CSV，自动分析、预测、预警
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime

from ai_forecast import train_and_forecast, explain_forecast
from ai_anomaly import detect_anomalies, LEVEL_EMOJI
from ai_insight import generate_insight, generate_suggestions

st.set_page_config(page_title="AI 运营数据分析", layout="wide", page_icon="🤖")

# ═══════════════════════════════════════════
# 侧边栏 — 数据上传与配置
# ═══════════════════════════════════════════

st.sidebar.header("📂 数据上传")

uploaded_file = st.sidebar.file_uploader("上传 CSV 文件", type=["csv"], help="支持任意 CSV 文件，自动识别列类型")

if uploaded_file is None:
    st.info("👈 请在左侧上传 CSV 数据文件开始分析")
    st.stop()

@st.cache_data
def load_uploaded(file):
    return pd.read_csv(file, encoding="utf-8-sig")

df_raw = load_uploaded(uploaded_file)

# ── 自动检测列类型 ──
def detect_columns(df):
    """自动识别日期列、数值列、分类列"""
    date_cols = []
    num_cols = []
    cat_cols = []

    for col in df.columns:
        # 跳过纯数值列（pd.to_datetime 会把数字也转成日期）
        if pd.api.types.is_numeric_dtype(df[col]):
            continue

        # 尝试日期
        try:
            s = pd.to_datetime(df[col], errors="coerce")
            if s.notna().sum() > len(df) * 0.8:
                date_cols.append(col)
                continue
        except (ValueError, TypeError):
            pass

        # 分类列
        if df[col].nunique() < 50:
            cat_cols.append(col)

    # 数值列
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 2]

    # 兜底：按列名常见模式检测日期
    date_name_patterns = ["date", "time", "日期", "时间", "dt", "day", "month", "year"]
    for col in df.columns:
        if col not in date_cols:
            col_lower = col.lower().replace("_", " ").replace("-", " ")
            if any(p in col_lower for p in date_name_patterns):
                try:
                    s = pd.to_datetime(df[col], errors="coerce")
                    if s.notna().sum() > len(df) * 0.5:
                        date_cols.append(col)
                        continue
                except Exception:
                    pass

    return date_cols, num_cols, cat_cols

date_cols, num_cols, cat_cols = detect_columns(df_raw)

st.sidebar.header("⚙️ 分析配置")

date_col = st.sidebar.selectbox("日期列", options=date_cols if date_cols else ["(无)"],
                                 help="用于时间序列分析和趋势图")
target_col = st.sidebar.selectbox("目标指标（用于预测）", options=num_cols if num_cols else ["(无)"],
                                   help="对该指标进行预测，通常是营收、销售额等")

group_cols = st.sidebar.multiselect("分组维度", options=cat_cols,
                                     default=cat_cols[:min(2, len(cat_cols))],
                                     help="用于饼图、柱状图等分组分析")

if date_col == "(无)" or target_col == "(无)":
    st.warning("请在上传的 CSV 中确保包含日期列和数值指标列")
    st.stop()

# ── 预处理 ──
df = df_raw.copy()
df["_date"] = pd.to_datetime(df[date_col])
df["_month"] = df["_date"].dt.to_period("M").astype(str)
df["_quarter"] = df["_date"].dt.quarter.apply(lambda x: f"Q{x}")
df["_week"] = df["_date"].dt.isocalendar().week.astype(int)

# 可选筛选器
if cat_cols:
    st.sidebar.header("🔍 筛选器")
    filters_active = {}
    for c in cat_cols[:4]:  # 最多显示4个筛选器
        vals = sorted(df[c].dropna().unique())
        if len(vals) <= 30:
            filters_active[c] = st.sidebar.multiselect(c, options=vals, default=vals)

    for c, vals in filters_active.items():
        if vals:
            df = df[df[c].isin(vals)]

if len(df) == 0:
    st.error("筛选后无数据，请调整筛选条件")
    st.stop()

# ═══════════════════════════════════════════
# 模型计算 (缓存)
# ═══════════════════════════════════════════

@st.cache_data(ttl=600)
def compute_weekly_data(_df, date_col_name, target_col_name):
    weekly = _df.groupby("_week").agg(
        **{target_col_name: (target_col_name, "sum"),
           "_count": (target_col_name, "count")}
    ).reset_index()
    weekly["_date"] = pd.date_range(
        start=_df["_date"].min(), periods=len(weekly), freq="W-MON"
    )
    return weekly

@st.cache_data(ttl=600)
def compute_forecast(_weekly_df, target_col_name, date_col_name):
    return train_and_forecast(_weekly_df, target_col_name, date_col_name, periods=8)

@st.cache_data(ttl=600)
def compute_anomalies(_weekly_df, _forecast_result, target_col_name):
    weekly = _weekly_df.copy()
    weekly["revenue"] = weekly[target_col_name]  # anomaly 模块用 revenue 命名
    weekly["orders"] = weekly["_count"]
    weekly["return_rate"] = 0  # 无退货数据时用0
    weekly["profit"] = weekly[target_col_name] * 0.3  # 估算
    weekly["profit_margin"] = 0.3
    weekly["date"] = weekly["_date"]
    return detect_anomalies(weekly, _forecast_result)

weekly_df = compute_weekly_data(df, date_col, target_col)
forecast_result = compute_forecast(weekly_df, target_col, "_date")
anomalies = compute_anomalies(weekly_df, forecast_result, target_col)

# ═══════════════════════════════════════════
# 标题
# ═══════════════════════════════════════════

st.title("🤖 AI 运营数据分析平台")
st.caption(f"已加载 {len(df):,} 条记录 | 日期列: {date_col} | 目标指标: {target_col} | 预测模型: {forecast_result['best_model']}")

# ═══════════════════════════════════════════
# KPI 指标卡片
# ═══════════════════════════════════════════

st.subheader("📊 核心指标")

total_target = df[target_col].sum()
avg_target = df[target_col].mean()
record_count = len(df)
date_span = f"{df['_date'].min().date()} ~ {df['_date'].max().date()}"

k1, k2, k3, k4 = st.columns(4)
# 尝试计算环比
if len(df) > 1:
    mid = len(df) // 2
    recent = df.iloc[mid:][target_col].sum()
    older = df.iloc[:mid][target_col].sum()
    delta = f"{(recent / older - 1) * 100:+.1f}%" if older > 0 else None
else:
    delta = None

with k1:
    st.metric(f"{target_col} 总计", f"{total_target:,.2f}", delta=delta)
with k2:
    st.metric("记录数", f"{record_count:,}")
with k3:
    st.metric(f"平均 {target_col}", f"{avg_target:,.2f}")
with k4:
    st.metric("数据跨度", date_span)

st.divider()

# ═══════════════════════════════════════════
# AI 运营洞察
# ═══════════════════════════════════════════

st.subheader("🧠 AI 运营洞察")

kpi_context = {
    "kpis": {
        f"{target_col} 总计": f"{total_target:,.2f}",
        "记录数": f"{record_count:,}",
        f"平均 {target_col}": f"{avg_target:,.2f}",
    },
}
if delta:
    kpi_context["changes"] = {f"{target_col} 环比": delta}

with st.spinner("AI 正在分析数据..."):
    insight = generate_insight(kpi_context, "review")

if isinstance(insight, dict):
    insight_col, alert_col = st.columns([3, 2])
    with insight_col:
        risk = insight.get("risk_level", "正常")
        risk_emoji = {"正常": "✅", "注意": "🟡", "警告": "🟠"}.get(risk, "✅")
        st.markdown(f"**{risk_emoji} 风险等级：{risk}**")
        st.info(insight.get("summary", "正在分析..."))
        if insight.get("insights"):
            st.markdown("**关键发现：**")
            for item in insight["insights"][:3]:
                st.markdown(f"- {item}")
    with alert_col:
        if anomalies:
            st.markdown("**🚨 预警信号：**")
            for a in anomalies[:5]:
                emoji = LEVEL_EMOJI.get(a["level"], "")
                st.warning(f"{emoji} **{a['metric']}** — {a['description']}")
        else:
            st.success("✅ 当前无异常预警")
else:
    st.info(str(insight))

st.divider()

# ═══════════════════════════════════════════
# 图表区（基于分组维度动态生成）
# ═══════════════════════════════════════════

if group_cols:
    left, right = st.columns([3, 2])

    with left:
        st.subheader(f"📈 月度 {target_col} 趋势")
        monthly = df.groupby("_month")[target_col].sum().reset_index()
        fig = px.bar(monthly, x="_month", y=target_col,
                     text=monthly[target_col].apply(lambda x: f"{x:,.0f}"),
                     color_discrete_sequence=["#4C78A8"])
        fig.update_layout(height=380, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("🥧 分组占比")
        if group_cols:
            primary_group = group_cols[0]
            cat_data = df.groupby(primary_group)[target_col].sum().reset_index()
            fig = px.pie(cat_data, values=target_col, names=primary_group,
                         color_discrete_sequence=px.colors.qualitative.Set2, hole=0.45)
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(height=380, margin=dict(t=10, b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # 第二行
    if len(group_cols) >= 2:
        left2, right2 = st.columns(2)
        with left2:
            st.subheader(f"📊 按 {group_cols[1]} 的 {target_col} 对比")
            g2_data = df.groupby(group_cols[1])[target_col].sum().sort_values(ascending=False).reset_index()
            fig = px.bar(g2_data, x=group_cols[1], y=target_col, color=target_col,
                         color_continuous_scale="Blues", text=g2_data[target_col].apply(lambda x: f"{x:,.0f}"))
            fig.update_layout(height=380, margin=dict(t=10, b=10), coloraxis_showscale=False)
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        with right2:
            st.subheader(f"📅 季度 {target_col} 变化")
            q_data = df.groupby("_quarter")[target_col].sum().reset_index()
            fig = px.bar(q_data, x="_quarter", y=target_col,
                         color_discrete_sequence=["#54A24B"],
                         text=q_data[target_col].apply(lambda x: f"{x:,.0f}"))
            fig.update_layout(height=380, margin=dict(t=10, b=10))
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

st.divider()

# ═══════════════════════════════════════════
# AI 销量预测面板
# ═══════════════════════════════════════════

st.subheader("🔮 预测：" + target_col)

fc_col1, fc_col2 = st.columns([3, 2])

with fc_col1:
    f_dates = forecast_result["future_dates"]
    f_vals = forecast_result["forecast"]
    f_low = forecast_result["conf_low"]
    f_high = forecast_result["conf_high"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=weekly_df["_date"], y=weekly_df[target_col],
        mode="lines", name="历史数据", line=dict(color="#4C78A8", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(f_dates), y=f_vals,
        mode="lines+markers", name="预测值", line=dict(color="#E45756", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(f_dates + f_dates[::-1]),
        y=np.array(f_high + f_low[::-1]),
        fill="toself", fillcolor="rgba(228,87,86,0.15)",
        line=dict(color="rgba(228,87,86,0)"), name="95% 置信区间",
    ))
    fig.update_layout(height=380, margin=dict(t=10, b=10),
                      xaxis_title="日期", yaxis_title=target_col,
                      legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True)

with fc_col2:
    st.markdown(f"**🏅 最优模型：{forecast_result['best_model']}**")
    st.markdown(f"RMSE: {forecast_result['metrics']['rmse']:,.2f}  |  MAE: {forecast_result['metrics']['mae']:,.2f}")

    st.markdown("**模型对比：**")
    comparison_df = pd.DataFrame(forecast_result["model_comparison"])
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    st.markdown("**未来预测：**")
    forecast_table = pd.DataFrame({
        "日期": f_dates,
        f"预测 {target_col}": [f"{v:,.2f}" for v in f_vals],
        "下限": [f"{v:,.2f}" for v in f_low],
        "上限": [f"{v:,.2f}" for v in f_high],
    })
    st.dataframe(forecast_table, use_container_width=True, hide_index=True)

    with st.spinner("AI 正在解读预测..."):
        explanation = explain_forecast(forecast_result)
    if isinstance(explanation, dict):
        st.info(explanation.get("summary", ""))
        for s in explanation.get("suggestions", [])[:3]:
            if isinstance(s, dict):
                st.markdown(f"- **{s.get('action', '')}** ({s.get('priority', '')}优先级)")
    else:
        st.caption(str(explanation))

st.divider()

# ═══════════════════════════════════════════
# AI 决策建议面板
# ═══════════════════════════════════════════

st.subheader("💡 AI 决策建议")

suggestions = generate_suggestions(anomalies, kpi_context["kpis"])

if suggestions:
    suggestion_cols = st.columns(min(len(suggestions), 3))
    for i, s in enumerate(suggestions[:6]):
        if isinstance(s, dict):
            with suggestion_cols[i % 3]:
                priority = s.get("priority", "中")
                color = "#e45756" if priority == "高" else ("#f58518" if priority == "中" else "#54a24b")
                st.markdown(f"""
                <div style="background:#fafafa;padding:12px;border-radius:8px;border-left:4px solid {color};margin:4px 0;">
                    <span style="background:{color};color:#fff;padding:2px 8px;border-radius:4px;font-size:12px;">{priority}优先级</span>
                    <p style="margin:8px 0 4px;"><strong>{s.get('action', '')}</strong></p>
                    <p style="font-size:12px;color:#666;">{s.get('reason', '')}</p>
                </div>
                """, unsafe_allow_html=True)
elif not anomalies:
    st.success("✅ 当前各项指标正常，继续保持常规运营节奏。")

st.divider()

# ═══════════════════════════════════════════
# 数据预览
# ═══════════════════════════════════════════

st.subheader("📋 数据预览")
st.dataframe(df.head(500), use_container_width=True, hide_index=True)

csv = df.to_csv(index=False, encoding="utf-8-sig")
st.download_button(
    label="📥 导出筛选数据 (CSV)",
    data=csv,
    file_name="filtered_data.csv",
    mime="text/csv",
)
