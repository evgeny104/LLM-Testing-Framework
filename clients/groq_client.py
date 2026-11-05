"""SUT-клиент: Groq играет роль тестируемого 'вендорского' LLM-эндпоинта.

Импортируется всеми тестами, которые делают реальный запрос к модели
(уровень 1 factual, уровень 3 hallucination).
"""
import os
from functools import lru_cache
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

@lru_cache(maxsize=1)
def get_client() -> Groq:
    """Create the Groq client on demand so offline tests can be collected."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to the project .env file.")
    return Groq(api_key=api_key)
