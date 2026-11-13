"""Уровень 5: недетерминированность и стабильность генерации.

Две проверки двух разных контрактов:
- temperature=0  → строго: все прогоны соответствуют эталону (детерминизм);
- temperature>0  → с допуском: не менее K из N прогонов соответствуют
  (законная недетерминированность, измеряем её долю).

Вызовы делаются прямо в теле тестов, без module-фикстур — здесь повторность
И есть предмет теста. RUNS мал (экономия токенов бесплатного тарифа).
"""
import os
import re
import pytest
from pydantic import BaseModel

from clients.groq_client import complete, complete_structured
from data.legal_corpus import get_article

pytestmark = [
    pytest.mark.stability,
    pytest.mark.llm_api,
    pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY is not set"),
]

QUESTION = "Сколько дней у меня есть на возврат товара?"
CONTEXT = get_article("art_18_zpp")   # Статья 18: возврат в течение 14 дней

RUNS = 3                  # число повторных прогонов (токены!)
POSITIVE_TEMP = 0.7       # «творческий» режим для проверки допуска
MIN_POSITIVE_PASS = 2     # допуск: минимум 2 из 3 ответов должны быть верны

SYSTEM_PROMPT = (
    f"Отвечай только на основе этого текста закона: {CONTEXT}. "
    "В ответе обязательно указывай номер статьи в формате 'Статья 18'."
)


class AnswerStats(BaseModel):
    """Схема структурированного ответа для проверки стабильности схемы."""
    days: int
    summary: str


def answer_is_correct(text: str) -> bool:
    """Эталонная проверка: ответ называет статью 18 и число 14."""
    cites_article = re.search(r"(?i)стать\w*\s*18", text)
    cites_days = re.search(r"\b14\b", text)
    return bool(cites_article and cites_days)


def test_zero_temperature_is_stable():
    """temperature=0: все прогоны обязаны соответствовать эталону."""
    failures = []
    for i in range(RUNS):
        answer = complete(system_content=SYSTEM_PROMPT, user_content=QUESTION)
        if not answer_is_correct(answer):
            failures.append((i, answer))
    assert not failures, (
        f"При temperature=0 {len(failures)}/{RUNS} ответов не по эталону: "
        f"{[(i, a[:80]) for i, a in failures]!r}"
    )


def test_positive_temperature_within_tolerance():
    """temperature>0: допуск — минимум MIN_POSITIVE_PASS из RUNS верных."""
    passed = 0
    wrong = []
    for i in range(RUNS):
        answer = complete(
            system_content=SYSTEM_PROMPT,
            user_content=QUESTION,
            temperature=POSITIVE_TEMP,
        )
        if answer_is_correct(answer):
            passed += 1
        else:
            wrong.append((i, answer))
    assert passed >= MIN_POSITIVE_PASS, (
        f"Только {passed}/{RUNS} ответов верны при t={POSITIVE_TEMP} "
        f"(допуск {MIN_POSITIVE_PASS}): {[(i, a[:80]) for i, a in wrong]!r}"
    )


def test_structured_output_stays_valid():
    """Структурированный вывод: схема валидна на каждом прогоне."""
    for _ in range(RUNS):
        result = complete_structured(
            AnswerStats,
            system_content=SYSTEM_PROMPT,
            user_content=QUESTION,
        )
        assert result.days == 14, f"days={result.days!r}, summary={result.summary!r}"
