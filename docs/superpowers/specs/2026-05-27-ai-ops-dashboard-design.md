# AI 运营数据分析平台 — 设计文档

**日期**: 2026-05-27
**项目**: Amazon Dashboard → AI 运营数据分析平台
**形态**: 增强版 Streamlit 看板

---

## 1. 项目概述

将现有的 Amazon 销售数据可视化看板改造为 AI 驱动的运营数据分析平台。在保留原有图表能力的基础上，新增 3 个 AI 引擎模块：销量预测、异常检测、LLM 洞察生成。

### 现有能力（保留）
- KPI 指标卡片（营收、订单数、客单价、退货率、利润率）
- 多维图表（月度趋势、类别占比、地区分布、退货率、Top10 产品、季度变化）
- 筛选器（日期范围、产品类别、地区）
- 数据明细表 + CSV 导出

### 新增能力
- AI 多模型销量预测 + 自动选最优 + 置信区间
- 多维度异常检测 + 分级预警
- LLM 生成运营洞察 + 可执行建议

---

## 2. 系统架构

```
Streamlit 前端 (app.py)
├── 现有: KPI 卡片 / 图表 / 数据表
├── 新增: AI 预测面板 / 预警面板 / 决策建议面板
│
├── ai_forecast.py   — 销量预测引擎
├── ai_anomaly.py    — 异常检测引擎
└── ai_insight.py    — AI 洞察引擎 (LLM)
```

---

## 3. 模块设计

### 3.1 ai_forecast.py — AI 销量预测引擎

**职责**: 自动训练多个时间序列模型，选择最优模型，生成预测值，调用 LLM 解读结果。

**函数接口**:

| 函数 | 输入 | 输出 |
|------|------|------|
| `train_and_forecast(df, target, periods)` | DataFrame, 目标列名, 预测周期数 | {forecast, conf_low, conf_high, best_model, metrics} |
| `compare_models(df, target)` | DataFrame, 目标列名 | [{model_name, rmse, mae, aic}] |
| `explain_forecast(forecast_result)` | 预测结果字典 | LLM 生成的中文解读 |

**模型池**: ARIMA, Prophet, LightGBM — 按 RMSE 自动选最优

**技术依赖**: statsmodels, prophet, lightgbm, scikit-learn

### 3.2 ai_anomaly.py — 异常检测引擎

**职责**: 对核心运营指标进行多维度统计检测，发现异常后分级预警。

**函数接口**:

| 函数 | 输入 | 输出 |
|------|------|------|
| `detect_anomalies(df, metrics)` | DataFrame, 指标列表 | [{metric, date, value, expected_range, level, description}] |
| `score_anomaly(value, baseline, method)` | 实际值, 基线, 方法名 | {score(0-100), level} |

**检测维度**:

| 指标 | 方法 | 阈值 |
|------|------|------|
| 营收 | 实际 vs 预测偏差 | >15% → 🟡, >25% → 🟠, >40% → 🔴 |
| 退货率 | 连续趋势 + σ 突破 | >均值+1σ → 🟡, +2σ → 🟠, +3σ → 🔴 |
| 订单量 | IQR 异常值 | < Q1-1.5IQR → 🟠 |
| 利润率 | 移动平均偏离 | < 4周均线90% → 🟡 |

**预警等级**: 🟡 注意 / 🟠 警告 / 🔴 严重

### 3.3 ai_insight.py — AI 洞察引擎

**职责**: 调用 LLM API，输入结构化数据上下文，生成自然语言洞察和可执行建议。

**函数接口**:

| 函数 | 输入 | 输出 |
|------|------|------|
| `generate_insight(context, type)` | 数据字典, "review"/"forecast"/"alert" | {summary, insights: [], suggestions: [], risk_level} |
| `generate_suggestions(anomalies, kpis)` | 异常列表 + KPI快照 | [{action, priority, reason}] |

**三种 Prompt 场景**:

| 场景 | 触发时机 | 输入数据 |
|------|---------|---------|
| 定期复盘 (review) | 看板加载时 | 本周 KPI + 环比变化 + Top5 产品 |
| 预测解读 (forecast) | 预测完成后 | 预测值 + 模型指标 + 历史趋势 |
| 异常响应 (alert) | 检测到异常时 | 异常详情 + 关联维度 + 历史对比 |

**LLM 配置**:
- 通过环境变量 `LLM_API_KEY` + `LLM_PROVIDER` 切换
- 默认使用 Claude (Anthropic SDK)
- Prompt 角色: "资深电商运营数据分析师"
- 输出格式: 结构化 JSON

---

## 4. 看板布局

单页滚动布局，从上到下：

1. **KPI 指标卡片** (保留)
2. **图表区 + AI 洞察卡片** (左: 原有图表, 右: 新增 AI 洞察)
3. **AI 预测面板 + 预警面板** (左右并列，新增)
4. **AI 决策建议面板** (新增，全宽)
5. **数据明细表** (保留)

---

## 5. 数据流

```
用户打开看板
  → load_data() 加载 CSV
  → 筛选器过滤数据
  → 现有图表渲染 (并行)
  → ai_forecast.train_and_forecast() 预测
  → ai_anomaly.detect_anomalies() 检测
  → ai_insight.generate_insight() 生成洞察 (依赖预测+异常结果)
  → 渲染 AI 面板
```

---

## 6. 依赖

```
streamlit, pandas, numpy, plotly          # 现有
statsmodels, prophet, lightgbm             # 预测
scikit-learn                               # 模型评估
anthropic                                  # LLM (Claude)
```

新增文件:
- `ai_forecast.py`
- `ai_anomaly.py`
- `ai_insight.py`
- 修改: `app.py`
- 更新: `requirements.txt`

---

## 7. 非目标（明确不做）

- 不含对话式聊天界面（用户选择了增强看板形态）
- 不含自动报告生成
- 不含实时数据流/数据库迁移（保持 CSV 文件读入）
- 不做用户登录/权限系统
