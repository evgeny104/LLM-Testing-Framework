"""Уровень 4: качество retrieval — RAGAS context precision и recall.

Метрики оценивают не ответ, а ПОИСК:
- context precision — все ли найденные статьи релевантны вопросу;
- context recall — вся ли эталонная информация покрыта найденными статьями.

top_k=3 намеренно с запасом: retriever тащит и «шумную» статью, метрики
это видят. Фикстура ragas_llm — из conftest.py.
"""
import os
import pytest
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics.collections import (
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
)

from clients.groq_client import complete
from clients.mock_retriever import retrieve
from data.legal_corpus import get_article

pytestmark = [
    pytest.mark.rag,
    pytest.mark.llm_api,
    pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY is not set"),
]

QUESTION = "Сколько дней у меня есть на возврат товара?"
# Эталонный ответ — «золотой» эталон для recall: что ВООБЩЕ должно быть в
# контексте, чтобы ответить на вопрос.
REFERENCE = "Потребитель вправе вернуть товар в течение 14 дней с момента покупки."

SYSTEM_PROMPT = (
    f"Отвечай только на основе этих статей закона:\n\n{get_article('art_18_zpp')}"
)

# Пороги учебные: keyword-retriever грубый, требуем умеренного качества.
PRECISION_THRESHOLD = 0.5
RECALL_THRESHOLD = 0.5


def test_context_precision_and_recall(ragas_llm):
    contexts = retrieve(QUESTION, top_k=3)

    response = complete(system_content=SYSTEM_PROMPT, user_content=QUESTION)

    sample = SingleTurnSample(
        user_input=QUESTION,
        response=response,
        retrieved_contexts=contexts,
        reference=REFERENCE,
    )
    dataset = EvaluationDataset(samples=[sample])

    result = evaluate(
        dataset=dataset,
        metrics=[
            LLMContextPrecisionWithoutReference(llm=ragas_llm),
            LLMContextRecall(llm=ragas_llm),
        ],
    )
    row = result.to_pandas().iloc[0]
    precision = row["llm_context_precision_without_reference"]
    recall = row["llm_context_recall"]

    assert precision >= PRECISION_THRESHOLD, (
        f"Context precision {precision} ниже порога {PRECISION_THRESHOLD}. "
        f"Найденные статьи: {[c[:50] for c in contexts]!r}"
    )
    assert recall >= RECALL_THRESHOLD, (
        f"Context recall {recall} ниже порога {RECALL_THRESHOLD}. "
        f"Найденные статьи: {[c[:50] for c in contexts]!r}"
    )
