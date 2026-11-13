"""Уровень 3: смысл ответа — DeepEval-метрика галлюцинаций.

HallucinationMetric просит судью (фикстура judge из conftest.py) оценить,
опирается ли ответ модели на переданный context. Всё, чего нет в context,
считается выдумкой. DeepEval 4.x: 1 = полностью опирается, 0 = сплошные
выдумки; тест падает, если score < threshold.
"""
import pytest
import os
from deepeval import assert_test
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase

from clients.groq_client import complete
from data.legal_corpus import get_article

pytestmark = [
    pytest.mark.llm_judge,
    pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY is not set"),
]


def test_hallucination_legal_context(judge):
    # Контекст — статья из базы знаний (тот же источник, что уровни 1 и 4).
    # Список строк: аналог retrieved chunks, можно расширить до нескольких.
    context = [get_article("art_18_zpp")]

    # Роль 1 — SUT: модель отвечает на вопрос строго по контексту.
    actual_output = complete(
        system_content=f"Отвечай только на основе этого текста закона: {context[0]}",
        user_content="Сколько дней у меня есть на возврат товара?",
    )

    # Роль 2 — оценка: судья сверяет ответ с контекстом.
    # Обе роли исполняет Groq, но это разные ответственности: SUT и оценщик.
    test_case = LLMTestCase(
        input="Сколько дней у меня есть на возврат товара?",
        actual_output=actual_output,
        context=context
    )
    # threshold=0.3 — учебный мягкий порог; на legal/fintech обычно 0.7–0.8.
    metric = HallucinationMetric(threshold=0.3, model=judge)
    assert_test(test_case, [metric])
