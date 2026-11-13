"""Уровень 3: LLM-as-judge через DeepEval.

AnswerRelevancyMetric — судья-модель (фикстура judge из conftest.py,
реализация — fixtures/judge_fixtures.py) оценивает, отвечает ли модель
по существу заданного вопроса.
Дороже и медленнее уровня 1, ловит семантику, а не подстроки.
"""
import os
import pytest
# assert_test — раннер DeepEval. Принимает LLMTestCase + список метрик,
# сам зовёт каждую метрику через judge и падает, если хотя бы одна < threshold.
from deepeval import assert_test

# AnswerRelevancyMetric — метрика "по делу ли ответ".
# Просит judge оценить, отвечает ли actual_output на input.
# Возвращает score в [0..1]; 1 = идеально, 0 = не по теме.
from deepeval.metrics import AnswerRelevancyMetric

# LLMTestCase — контейнер данных для одного тест-кейса LLM.
# Поля: input (вопрос), actual_output (ответ модели),
# context/retrieval_context/expected_output (для других метрик).
# Метрики читают только те поля, которые им нужны.
from deepeval.test_case import LLMTestCase

# module-level маркер: все тесты этого файла получают @pytest.mark.llm_judge.
# Позволяет фильтровать через `pytest -m llm_judge` / `-m "not llm_judge"`.
pytestmark = [
    pytest.mark.llm_judge,
    pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY is not set"),
]


def test_answer_relevancy(judge):
    # LLMTestCase собирает данные, которые метрика будет оценивать.
    # Тут нужны только input + actual_output — AnswerRelevancy другие поля не читает.
    test_case = LLMTestCase(
        input="Столица Франции?",
        actual_output="Столица Франции — Париж."
    )
    # threshold=1 — жёстко: score должен быть ровно 1, иначе assert_test упадёт.
    # model=judge — какой моделью-судьёй мерить (наша GroqJudge вместо OpenAI-дефолта).
    metric = AnswerRelevancyMetric(threshold=1, model=judge)
    # запуск: assert_test дёргает metric.measure(test_case), сравнивает score с
    # threshold.
    assert_test(test_case, [metric])
