"""Уровень 7: AI-агент — модель + инструменты (Skills/Tools).

Агент решает, какой инструмент вызвать; исполняет вызов приложение.
Инструменты — реестр TOOLS: имя → функция. Решение о вызове модель
отдаёт через complete_structured (tool-calling на gpt-oss-20b ломан,
структурированный вывод — нет).

Интеграция через MCP (внешний сервер инструментов) — в roadmap; здесь
тот же контракт «модель выбирает инструмент → приложение исполняет»
проверяется локально.
"""
import os
import re
import pytest
from pydantic import BaseModel, Field

from clients.groq_client import complete, complete_structured
from clients.mock_retriever import retrieve

pytestmark = [
    pytest.mark.agents,
    pytest.mark.llm_api,
    pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY is not set"),
]

QUESTION = "Сколько дней у меня есть на возврат товара?"


class ToolCall(BaseModel):
    """Решение агента: какой инструмент вызвать и с каким аргументом."""
    tool: str = Field(description="Имя инструмента: 'retrieve' или 'answer'")
    question: str = Field(
        description="Для retrieve — поисковый запрос; для answer — итоговый ответ."
    )


# Реестр инструментов агента (аног Skills/Tools): имя → функция.
TOOLS = {
    "retrieve": retrieve,
}

AGENT_SYSTEM_PROMPT = (
    "Ты агент юридического ассистента с инструментом 'retrieve' — поиск "
    "статей по базе знаний. Если для ответа нужен текст закона — вызови "
    "retrieve с поисковым запросом. Иначе ответь сам через инструмент "
    "'answer'. Ответ давай строго в формате JSON."
)


def test_agent_chooses_retrieve_tool():
    """Модель осознанно выбирает инструмент retrieve для юридического вопроса."""
    decision = complete_structured(
        ToolCall,
        system_content=AGENT_SYSTEM_PROMPT,
        user_content=QUESTION,
    )
    assert decision.tool == "retrieve", (
        f"Агент выбрал инструмент {decision.tool!r} вместо 'retrieve'"
    )
    assert decision.question.strip(), "Пустой поисковый запрос для retrieve"


def test_agent_loop_answers_with_tool_result():
    """Полный цикл агента: решение → вызов инструмента → ответ по результату.

    Исполнение инструмента — за приложением (здесь — тест), не за моделью.
    """
    decision = complete_structured(
        ToolCall,
        system_content=AGENT_SYSTEM_PROMPT,
        user_content=QUESTION,
    )
    assert decision.tool in TOOLS, f"Неизвестный инструмент: {decision.tool!r}"

    # Приложение исполняет выбранный инструмент:
    contexts = TOOLS[decision.tool](decision.question, top_k=2)

    answer = complete(
        system_content=(
            "Отвечай только на основе этих статей закона:\n\n"
            + "\n\n".join(contexts)
        ),
        user_content=QUESTION,
    )
    cites_article = re.search(r"(?i)стать\w*\s*18", answer)
    cites_days = re.search(r"\b14\b", answer)
    assert cites_article and cites_days, (
        f"Ответ агента не по эталону: {answer[:120]!r}"
    )
