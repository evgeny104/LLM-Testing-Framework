"""Уровень 1: детерминированные проверки на ОДНОМ захардкоженном кейсе.

Один вызов Groq в фикстуре module-scope → несколько regex-шаблонов
через parametrize. Учебная база: temperature=0 для воспроизводимости,
system-промпт с директивой формата цитаты.
"""
import os
import re
import pytest
from clients.groq_client import complete

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


@pytest.fixture(scope="module")
def model_answer():
    answer = complete(
        system_content=(
            f"Отвечай только на основе этого текста закона: {CONTEXT}. "
            "В ответе обязательно указывай номер статьи в формате 'Статья 15'."
        ),
        user_content=QUESTION,
    )
    print(answer)
    return answer


@pytest.mark.parametrize("pattern, description", [
    (r"(?i)стать\w*\s*15", "цитата статьи"),
    (r"\b14\b", "число дней 14"),
    (r"(?i)возврат|верн\w*|возвращ\w*", "слово 'возврат'")
])
def test_answer_matches_pattern(model_answer, pattern, description):
    assert re.search(pattern, model_answer), (
        f"В ответе не найдено: {description}. Ответ: {model_answer!r}"
    )
