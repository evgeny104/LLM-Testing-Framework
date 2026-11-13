"""Уровень 2: валидация структурированного вывода по pydantic-схеме.

Модель обязана ответить JSON строго по схеме LegalAnswer. instructor
(JSON-режим — tool-calling ломается на openai/gpt-oss-20b, см.
evaluation/judge_model.py) парсит ответ и сам гоняет его через pydantic:
если модель ошиблась в схеме — instructor повторяет запрос с подсказкой.

Контекст — статья art_18_zpp из базы знаний, тот же источник, что у
уровня 1. Проверки детерминированные: сравнение полей с эталоном.
"""
import os
import pytest
from pydantic import BaseModel

from clients.groq_client import complete_structured
from data.legal_corpus import get_article

pytestmark = [
    pytest.mark.structured,
    pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY is not set"),
]


class LegalAnswer(BaseModel):
    """Схема структурированного ответа юридического ассистента."""
    article: int      # номер статьи закона
    days: int         # число дней из статьи
    summary: str      # краткий вывод одним предложением


QUESTION = "Сколько дней у меня есть на возврат товара?"
CONTEXT = get_article("art_18_zpp")   # Статья 18: возврат товара в течение 14 дней
EXPECTED_ARTICLE = 18
EXPECTED_DAYS = 14


@pytest.fixture(scope="module")
def structured_answer():
    """Один вызов Groq на файл: модель отвечает JSON по схеме LegalAnswer."""
    return complete_structured(
        LegalAnswer,
        system_content=(
            f"Отвечай только на основе этого текста закона: {CONTEXT}. "
            "Ответ дай строго в формате JSON."
        ),
        user_content=QUESTION,
    )


def test_article_matches_etalon(structured_answer):
    assert structured_answer.article == EXPECTED_ARTICLE


def test_days_match_etalon(structured_answer):
    assert structured_answer.days == EXPECTED_DAYS


def test_summary_not_empty(structured_answer):
    assert structured_answer.summary.strip(), (
        f"Пустой summary: {structured_answer.summary!r}"
    )
