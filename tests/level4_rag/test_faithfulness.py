"""
Уровень 4 — RAGAS Faithfulness. Проверяет, что ответ модели опирается только на
retrieval_context, а не на внутренние знания LLM.

Новое поколение метрик (ragas.metrics.collections) оценивает sample напрямую
через metric.score(...) → MetricResult.value; старый ragas.evaluate() с ними
не совместим. Фикстура ragas_llm — из conftest.py.
"""

import pytest
import os
from ragas.metrics.collections import Faithfulness

from core.llm import complete
from vectorstore import retrieve

pytestmark = [
    pytest.mark.rag,
    pytest.mark.llm_api,
    pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY is not set"),
]


def test_rag_wrapper_builds(ragas_llm):
    assert ragas_llm is not None


def test_faithfulness_minimal(ragas_llm):
    question = "Сколько дней у меня есть на возврат товара?"
    contexts = retrieve(question, top_k=2)

    answer = complete(
        system_content=(
            "Отвечай только на основе этих статей закона:\n\n"
            + "\n\n".join(contexts)
        ),
        user_content=question,
    )

    result = Faithfulness(llm=ragas_llm).score(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
    )
    score = result.value
    assert score >= 0.5, f"Faithfulness score {score} below threshold 0.5"
