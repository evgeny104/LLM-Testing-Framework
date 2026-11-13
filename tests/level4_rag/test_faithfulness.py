"""
Уровень 4 — RAGAS Faithfulness. Проверяет, что ответ модели опирается только на
retrieval_context, а не на внутренние знания LLM.

Фикстура ragas_llm — из conftest.py (реализация — fixtures/judge_fixtures.py).
"""

import pytest
import os
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import Faithfulness

from clients.groq_client import complete
from clients.mock_retriever import retrieve

pytestmark = [
    pytest.mark.rag,
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

    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
    )
    dataset = EvaluationDataset(samples=[sample])

    result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(llm=ragas_llm)],
    )

    score = result.to_pandas()["faithfulness"].iloc[0]
    assert score >= 0.5, f"Faithfulness score {score} below threshold 0.5"
