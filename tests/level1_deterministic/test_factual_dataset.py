"""Уровень 1: детерминированные проверки на МИНИ-DATASET из нескольких кейсов.

Фикстура model_answer — из фабрики fixtures.llm_fixtures.make_model_answer
(параметризована по CASES из golden_cases.py): один вызов Groq на кейс,
тесты строят regex динамически из полей кейса.
"""
import os
import re
import pytest
from fixtures.llm_fixtures import make_model_answer
from tests.level1_deterministic.golden_cases import CASES

pytestmark = [
    pytest.mark.llm_api,
    pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY is not set"),
]

model_answer = make_model_answer(CASES)


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
