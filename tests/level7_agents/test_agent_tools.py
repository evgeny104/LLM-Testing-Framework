"""Уровень 7: AI-агент — модель + инструменты через нативный tool-calling.

Цикл вызова инструмента (исполняет приложение, не модель):
1. модель получает вопрос + объявление инструмента → отвечает tool_call;
2. приложение выполняет вызванный инструмент (здесь — тест);
3. результат возвращается модели → финальный ответ.

Инструмент — retrieve: поиск статей по базе знаний.
MCP-интеграция (внешний сервер инструментов) — в roadmap; здесь тот же
контракт проверяется локальным инструментом.
"""
import json
import os
import re
import pytest

from core.agent import AGENT_SYSTEM_PROMPT, TOOLS, ask_agent

pytestmark = [
    pytest.mark.agents,
    pytest.mark.llm_api,
    pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY is not set"),
]

QUESTION = "Сколько дней у меня есть на возврат товара?"

AGENT_MESSAGES = [
    {"role": "system", "content": AGENT_SYSTEM_PROMPT},
    {"role": "user", "content": QUESTION},
]


def test_agent_calls_retrieve_tool():
    """Модель сама решает вызвать retrieve и передаёт осмысленный запрос."""
    response = ask_agent(AGENT_MESSAGES)
    tool_calls = response.choices[0].message.tool_calls
    assert tool_calls, "Модель не вызвала инструмент, хотя вопрос требует поиска"
    call = tool_calls[0]
    assert call.function.name == "retrieve"
    args = json.loads(call.function.arguments)
    assert args["query"].strip(), "Пустой поисковый запрос"


def test_agent_loop_answers_with_tool_result():
    """Полный цикл: tool_call → приложение исполняет → модель отвечает по результату."""
    messages = list(AGENT_MESSAGES)

    # Шаг 1: модель просит вызвать инструмент.
    first = ask_agent(messages)
    message = first.choices[0].message
    assert message.tool_calls, (
        f"Модель не вызвала инструмент, ответила: {str(message.content)[:100]!r}"
    )
    tool_call = message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)

    # Шаг 2: приложение исполняет инструмент (модель это делать не умеет —
    # она только текст; исполнение — всегда код через реестр TOOLS).
    assert tool_call.function.name in TOOLS, f"Неизвестный инструмент: {tool_call.function.name!r}"
    contexts = TOOLS[tool_call.function.name](args["query"], top_k=2)
    tool_result = "\n\n".join(contexts)

    # Шаг 3: результат уходит модели в сообщении role="tool".
    messages.append(first.choices[0].message)
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": tool_result,
    })
    final = ask_agent(messages)
    answer = final.choices[0].message.content

    cites_article = re.search(r"(?i)стать\w*\s*18", answer)
    cites_days = re.search(r"\b14\b", answer)
    assert cites_article and cites_days, (
        f"Ответ агента не по эталону: {answer[:120]!r}"
    )
