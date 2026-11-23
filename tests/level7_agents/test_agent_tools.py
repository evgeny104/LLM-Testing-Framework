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

from clients.groq_client import get_client, MODEL
from clients.vector_store import retrieve

pytestmark = [
    pytest.mark.agents,
    pytest.mark.llm_api,
    pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY is not set"),
]

QUESTION = "Сколько дней у меня есть на возврат товара?"

# Реальный паттерн RAG-агентов: system-промпт ОБЯЗЫВАЕТ искать в базе.
# Без этого модель отвечает из памяти обучения (проверено: игнорирует
# инструмент при tool_choice="auto").
AGENT_SYSTEM_PROMPT = (
    "Ты юридический ассистент. Перед ответом на любой вопрос о законе "
    "обязательно ищи в базе знаний через инструмент retrieve. Отвечай "
    "только на основе найденных статей."
)

# Объявление инструмента для модели: имя, описание, схема аргументов.
RETRIEVE_TOOL = {
    "type": "function",
    "function": {
        "name": "retrieve",
        "description": "Поиск релевантных статей закона по базе знаний",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Поисковый запрос"},
            },
            "required": ["query"],
        },
    },
}


def ask_agent(messages):
    """Один шаг диалога с агентом в режиме tool-calling."""
    return get_client().chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=[RETRIEVE_TOOL],
        tool_choice="auto",
        temperature=0,
    )


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
    # она только текст; исполнение — всегда код).
    contexts = retrieve(args["query"], top_k=2)
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
