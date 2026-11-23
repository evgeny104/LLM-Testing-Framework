"""Уровень 4: качество retrieval — RAGAS context precision и recall.

Метрики оценивают не ответ, а ПОИСК:
- context precision — все ли найденные статьи релевантны вопросу;
- context recall — вся ли эталонная информация покрыта найденными статьями.

top_k=3 намеренно с запасом: retriever тащит и менее релевантные статьи,
метрики это видят. Новое API метрик: metric.score(...) → MetricResult.value.
Фикстура ragas_llm — из conftest.py.
"""
import os
import pytest
from ragas.metrics.collections import ContextPrecisionWithoutReference, ContextRecall

from core.llm import complete
from vectorstore import retrieve

pytestmark = [
    pytest.mark.rag,
    pytest.mark.llm_api,
    pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY is not set"),
]

QUESTION = "Сколько дней у меня есть на возврат товара?"
# Эталонный ответ — «золотой» эталон для recall: что ВООБЩЕ должно быть в
# контексте, чтобы ответить на вопрос.
REFERENCE = "Потребитель вправе вернуть товар в течение 14 дней с момента покупки."

# Пороги учебные, с запасом: оценка судьёй недетерминирована — score может
# плавать около порога от прогона к прогону (тема 4). Для жёстких контрактов
# retrieval пороги поднимают после сбора статистики прогонов.
PRECISION_THRESHOLD = 0.3
RECALL_THRESHOLD = 0.3


def test_context_precision_and_recall(ragas_llm):
    contexts = retrieve(QUESTION, top_k=3)

    response = complete(
        system_content=(
            "Отвечай только на основе этих статей закона:\n\n"
            + "\n\n".join(contexts)
        ),
        user_content=QUESTION,
    )

    precision = ContextPrecisionWithoutReference(llm=ragas_llm).score(
        user_input=QUESTION,
        response=response,
        retrieved_contexts=contexts,
    ).value
    recall = ContextRecall(llm=ragas_llm).score(
        user_input=QUESTION,
        reference=REFERENCE,
        retrieved_contexts=contexts,
    ).value

    assert precision >= PRECISION_THRESHOLD, (
        f"Context precision {precision} ниже порога {PRECISION_THRESHOLD}"
    )
    assert recall >= RECALL_THRESHOLD, (
        f"Context recall {recall} ниже порога {RECALL_THRESHOLD}"
    )
