"""AI 洞察引擎 — 支持阿里云百炼 / Anthropic / OpenAI 兼容 API"""
import os
import json

# 默认配置
DEFAULT_API_KEY = "sk-3559232492404cf08e93aeb87cb761d2"
DEFAULT_MODEL = "qwen3.5-omni-plus"
BAILIAN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def _get_llm_client():
    """根据环境变量初始化 LLM 客户端"""
    provider = os.getenv("LLM_PROVIDER", "bailian").lower()
    api_key = os.getenv("LLM_API_KEY", DEFAULT_API_KEY)

    model = os.getenv("LLM_MODEL", DEFAULT_MODEL)

    if provider in ("bailian", "openai", "openai_compatible"):
        from openai import OpenAI
        base_url = os.getenv("LLM_BASE_URL", BAILIAN_BASE_URL)
        return OpenAI(api_key=api_key, base_url=base_url), model

    elif provider == "anthropic":
        from anthropic import Anthropic
        return Anthropic(api_key=api_key), model

    return None, None


def _call_llm(system_prompt, user_prompt):
    """调用 LLM，自动适配不同 API 格式"""
    client, model = _get_llm_client()

    if client is None:
        return _fallback_insight(user_prompt)

    provider = os.getenv("LLM_PROVIDER", "bailian").lower()

    try:
        if provider == "anthropic":
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = response.content[0].text.strip()
        else:
            # OpenAI 兼容格式 (百炼 / OpenAI)
            response = client.chat.completions.create(
                model=model,
                max_tokens=1024,
                temperature=0.7,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            text = response.choices[0].message.content.strip()

        # 解析 JSON
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        return json.loads(text)
    except Exception as e:
        return {"summary": f"AI 调用失败: {e}", "insights": [], "suggestions": [], "risk_level": "正常"}


def _fallback_insight(context_str):
    return "（无 AI 密钥配置）请设置 LLM_API_KEY 环境变量以启用 AI 洞察。"


SYSTEM_PROMPT = """你是一个资深的数据分析师，专注于业务运营数据分析。

你的职责：
1. 分析给定的运营数据，发现关键趋势和问题
2. 用简洁专业的中文给出洞察
3. 提出具体可执行的运营建议

输出格式要求（严格JSON）：
{
  "summary": "一段话总结核心发现（100字以内）",
  "insights": ["发现1", "发现2", "发现3"],
  "suggestions": [
    {"action": "具体行动", "priority": "高/中/低", "reason": "原因"},
    ...
  ],
  "risk_level": "正常/注意/警告"
}

规则：
- 不要编造数据中没有的数字
- 建议要具体可执行，不能是"加强管理"这种空话
- 如果有异常，明确指出来并给出应对措施
"""


def generate_insight(context, insight_type="review"):
    """生成 AI 运营洞察。"""
    if insight_type == "review":
        user_prompt = f"""请对以下运营数据进行复盘分析：

当前KPI数据：
{json.dumps(context.get('kpis', {}), ensure_ascii=False, indent=2)}

环比变化：
{json.dumps(context.get('changes', {}), ensure_ascii=False, indent=2)}

请给出运营复盘总结。"""
    elif insight_type == "forecast":
        user_prompt = f"""请解读以下预测结果：

最优模型：{context.get('best_model', 'Unknown')}
模型RMSE：{context.get('rmse', 'N/A')}
未来预测值：{json.dumps(dict(zip(context.get('future_dates', []), context.get('forecast', []))), ensure_ascii=False)}
模型排名：{context.get('model_ranking', [])}

请解读预测趋势，给出运营建议。"""
    elif insight_type == "alert":
        user_prompt = f"""以下异常已被检测到，请分析并给出应对建议：

异常列表：
{json.dumps(context.get('anomalies', []), ensure_ascii=False, indent=2)}

当前KPI：
{json.dumps(context.get('kpis', {}), ensure_ascii=False, indent=2)}

请分析异常原因，给出具体的应对措施。"""
    else:
        user_prompt = f"请分析以下数据：{json.dumps(context, ensure_ascii=False)}"

    return _call_llm(SYSTEM_PROMPT, user_prompt)


def generate_suggestions(anomalies, kpis):
    """基于异常和 KPI 生成可执行的运营建议。"""
    if not anomalies:
        return [{"action": "当前各项指标正常，继续保持常规运营节奏",
                 "priority": "低", "reason": "无异常检测"}]

    context = {"anomalies": anomalies, "kpis": kpis}
    result = generate_insight(context, "alert")
    if isinstance(result, dict):
        return result.get("suggestions", [])
    return []
