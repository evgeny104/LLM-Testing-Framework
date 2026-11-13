"""Уровень 3: смысл ответа — DeepEval-метрика релевантности.

AnswerRelevancyMetric просит судью (фикстура judge из conftest.py) оценить,
отвечает ли actual_output по существу на input. Score в [0..1], падает,
если score < threshold.
"""
import os
import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

pytestmark = [
    pytest.mark.llm_judge,
    pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY is not set"),
]


def test_answer_relevancy(judge):
    test_case = LLMTestCase(
        input="Столица Франции?",
        actual_output="Столица Франции — Париж."
    )
    # threshold=1 — жёстко: ответ должен быть полностью релевантным.
    metric = AnswerRelevancyMetric(threshold=1, model=judge)
    assert_test(test_case, [metric])
