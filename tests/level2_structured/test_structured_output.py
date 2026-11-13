"""Уровень 2: валидация структурированного вывода по pydantic-схеме.

Модель обязана ответить JSON строго по схеме LegalAnswer (instructor в JSON-режиме,
т.к. tool-calling ломается на openai/gpt-oss-20b — см. evaluation/judge_model.py).
Проверки детерминированные: сверяем поля объекта с эталоном, без regex.
"""
import os
import pytest
from pydantic import BaseModel

pytestmark = [
    pytest.mark.structured,
    pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY is not set"),
]


class LegalAnswer(BaseModel):
    """Схема структурированного ответа юридического ассистента."""
    article: int      # номер статьи закона
    days: int         # число дней из статьи
    summary: str      # краткий вывод одним предложением
