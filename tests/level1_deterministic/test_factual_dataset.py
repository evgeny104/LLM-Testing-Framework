"""Уровень 1: детерминированные проверки на МИНИ-DATASET из нескольких кейсов.

Параметризованная фикстура `model_answer` пробегает по CASES (golden_cases.py),
для каждого делает вызов Groq и возвращает dict с ответом + эталонами.
Тесты строят regex ДИНАМИЧЕСКИ через f-string из полей кейса.
"""
import os
import re
import pytest
from clients.groq_client import complete
from tests.level1_deterministic.golden_cases import CASES

pytestmark = [
    pytest.mark.llm_api,
    pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY is not set"),
]


@pytest.fixture(scope="module", params=CASES, ids=[c["id"] for c in CASES])
def model_answer(request):
    case = request.param
    answer = complete(
        system_content=(
            f"Отвечай только на основе этого текста закона: {case['context']}. "
            f"В ответе обязательно указывай номер статьи в формате "
            f"'Статья {case['expected_article']}'."
        ),
        user_content=case["question"],
    )
    return {**case, "answer": answer}


def test_answer_cites_expected_article(model_answer):
    pattern = rf"(?i)стать\w*\s*{model_answer['expected_article']}"
    assert re.search(pattern, model_answer["answer"]), (
        f"Не найдена цитата 'Статья {model_answer['expected_article']}' "
        f"в ответе: {model_answer['answer']!r}"
    )


def test_answer_cites_expected_days(model_answer):
    pattern = rf"\b{model_answer['expected_days']}\b"
    assert re.search(pattern, model_answer["answer"]), (
        f"Не найдено число {model_answer['expected_days']} "
        f"в ответе: {model_answer['answer']!r}"
    )


def test_answer_not_empty(model_answer):
    assert model_answer["answer"].strip(), (
        f"Ответ пустой: {model_answer['answer']!r}"
    )
