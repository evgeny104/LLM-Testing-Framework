"""Уровень 3: LLM-as-judge — проверка галлюцинаций.

HallucinationMetric просит судью (фикстура judge из conftest.py) оценить,
выходил ли ответ за пределы переданного контекста (`context`).
DeepEval 4.x: 1 = хорошо, 0 = плохо, threshold — минимум.
"""
import pytest
import os
from deepeval import assert_test
# HallucinationMetric — сверяет actual_output с полем context.
# Всё в ответе, чего НЕТ в context, judge считает галлюцинацией.
# DeepEval 4.x: 1 = ответ полностью опирается на context, 0 = сплошные выдумки.
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase
# complete — SUT-клиент (Groq в роли тестируемого "вендора"). ВАЖНО: тут два
# разных вызова модели:
#   1) через complete → генерируем actual_output (роль SUT),
#   2) внутри HallucinationMetric → judge оценивает ответ (роль оценщика).
# Обе роли исполняет Groq, но концептуально это разные ответственности.
from clients.groq_client import complete
from data.legal_corpus import get_article

pytestmark = [
    pytest.mark.llm_judge,
    pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY is not set"),
]


def test_hallucination_legal_context(judge):
    # context — статья из базы знаний (тот же источник, что у golden dataset
    # и RAG-тестов). СПИСОК строк, не строка: DeepEval поддерживает несколько
    # кусков контекста (аналог retrieved chunks в RAG) — можно расширить.
    context = [get_article("art_18_zpp")]

    # === Роль 1: SUT. Просим "вендорскую" модель ответить на юзер-вопрос. ===
    # В system-промпт кладём context[0] — единственный кусок; в реальном RAG
    # сюда уйдёт объединение всех retrieved docs.
    actual_output = complete(
        system_content=f"Отвечай только на основе этого текста закона: {context[0]}",
        user_content="Сколько дней у меня есть на возврат товара?",
    )

    # === Роль 2: сборка данных для оценки. ===
    # LLMTestCase несёт три поля, которые нужны HallucinationMetric:
    # input — вопрос, actual_output — что модель ответила, context — эталонная
    # "истина".
    test_case = LLMTestCase(
        input="Сколько дней у меня есть на возврат товара?",
        actual_output=actual_output,
        context=context
    )
    # threshold=0.3 — МЯГКИЙ порог: "минимум 30% фактов должны опираться на
    # context". На legal/fintech обычно 0.7–0.8 (строже). Тут пока учебный кейс.
    metric = HallucinationMetric(threshold=0.3, model=judge)
    # assert_test → metric.measure(test_case) → judge получает промпт вида
    # "вот ответ [actual_output], вот эталон [context], оцени долю фактов от
    # context" → judge возвращает JSON {score, reason} → метрика сверяет score
    # с threshold.
    assert_test(test_case, [metric])
