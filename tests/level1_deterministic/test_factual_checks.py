"""Уровень 1: детерминированные проверки на ОДНОМ захардкоженном кейсе.

Фикстура model_answer — из фабрики fixtures.llm_fixtures.make_single_answer:
один вызов Groq на файл (module-scope), ответ проверяется несколькими
regex-шаблонами через parametrize. temperature=0 — воспроизводимость.
"""
import os
import re
import pytest
from fixtures.llm_fixtures import make_single_answer

pytestmark = [
    pytest.mark.llm_api,
    pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY is not set"),
]

QUESTION = "Сколько дней у меня есть на возврат товара?"
CONTEXT = (
    "Статья 15 закона о защите прав потребителей: "
    "потребитель имеет право вернуть товар в течение 14 дней "
    "с момента покупки без объяснения причин."
)

model_answer = make_single_answer(
    system_content=(
        f"Отвечай только на основе этого текста закона: {CONTEXT}. "
        "В ответе обязательно указывай номер статьи в формате 'Статья 15'."
    ),
    question=QUESTION,
)


@pytest.mark.parametrize("pattern, description", [
    (r"(?i)стать\w*\s*15", "цитата статьи"),
    (r"\b14\b", "число дней 14"),
    (r"(?i)возврат|верн\w*|возвращ\w*", "слово 'возврат'")
])
def test_answer_matches_pattern(model_answer, pattern, description):
    assert re.search(pattern, model_answer), (
        f"В ответе не найдено: {description}. Ответ: {model_answer!r}"
    )
