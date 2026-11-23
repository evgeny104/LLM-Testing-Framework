"""Фикстуры LLM-обёртки для RAGAS-метрик (уровень 4)."""
import os
import pytest
from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.llms import llm_factory

from core.llm import GROQ_BASE_URL, MODEL

load_dotenv()


@pytest.fixture(scope="module")
def ragas_llm():
    """LLM для RAGAS-метрик.

    Метрики нового поколения (ragas.metrics.collections) требуют InstructorLLM
    через llm_factory и внутри работают асинхронно — нужен AsyncOpenAI-клиент
    (синхронный даст TypeError: use generate() instead).
    """
    client = AsyncOpenAI(
        api_key=os.getenv("GROQ_API_KEY", "missing-api-key"),
        base_url=GROQ_BASE_URL,
    )
    # reasoning_effort="low": судья отвечает коротким JSON — без этого reasoning-
    # модель жжёт токены на рассуждения и генерация обрывается (IncompleteOutput).
    return llm_factory(MODEL, client=client, reasoning_effort="low", max_tokens=4096)
