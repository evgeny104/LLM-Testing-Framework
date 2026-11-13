"""Уровень 1: детерминированные проверки на ОДНОМ кейсе из базы знаний.

Фикстура model_answer — из фабрики fixtures.llm_fixtures.make_single_answer:
один вызов Groq на файл (module-scope), ответ проверяется несколькими
regex-шаблонами через parametrize. Контекст — статья art_18_zpp из
data/legal_corpus.py, тот же источник, что у golden dataset и RAG-тестов.
"""
import os
import re
import pytest
from fixtures.llm_fixtures import make_single_answer
from data.legal_corpus import get_article

pytestmark = [
    pytest.mark.llm_api,
    pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY is not set"),
]

QUESTION = "Сколько дней у меня есть на возврат товара?"
CONTEXT = get_article("art_18_zpp")   # Статья 18: возврат товара в течение 14 дней

model_answer = make_single_answer(
    system_content=(
        f"Отвечай только на основе этого текста закона: {CONTEXT}. "
        "В ответе обязательно указывай номер статьи в формате 'Статья 18'."
    ),
    question=QUESTION,
)


@pytest.mark.parametrize("pattern, description", [
    (r"(?i)стать\w*\s*18", "цитата статьи"),
    (r"\b14\b", "число дней 14"),
    (r"(?i)возврат|верн\w*|возвращ\w*", "слово 'возврат'")
])
def test_answer_matches_pattern(model_answer, pattern, description):
    assert re.search(pattern, model_answer), (
        f"В ответе не найдено: {description}. Ответ: {model_answer!r}"
    )
