"""Фикстуры LLM-обёртки для RAGAS-метрик (уровень 4)."""
import os
import pytest
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

from clients.groq_client import MODEL

load_dotenv()


@pytest.fixture(scope="module")
def ragas_llm():
    """LLM-обёртка для RAGAS-метрик: Groq через интерфейс langchain."""
    groq_llm = ChatOpenAI(
        model=MODEL,
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY", "missing-api-key"),
        temperature=0,
        max_tokens=4096,
        # reasoning_effort="low": RAGAS-judge отвечает коротко — экономим токены
        # бесплатного тарифа и снижаем риск обрыва генерации.
        model_kwargs={"reasoning_effort": "low"},
    )
    return LangchainLLMWrapper(groq_llm)
