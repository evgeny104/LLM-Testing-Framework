"""SUT-клиент: Groq играет роль тестируемого 'вендорского' LLM-эндпоинта.

Импортируется всеми тестами, которые делают реальный запрос к модели
(уровень 1 factual, уровень 3 hallucination, уровень 4 RAG).
"""
import os
from functools import lru_cache
from dotenv import load_dotenv
from groq import Groq
from groq.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

load_dotenv()

MODEL = "openai/gpt-oss-20b"


@lru_cache(maxsize=1)
def get_client() -> Groq:
    """Create the Groq client on demand so offline tests can be collected."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to the project .env file.")
    return Groq(api_key=api_key)


def complete(system_content: str, user_content: str, temperature: float = 0) -> str:
    """Один вызов chat.completions: собирает messages и возвращает текст ответа.

    Общая точка для всех тестовых фикстур — убирает дублирование конструирования
    сообщений и распаковки choices[0]. temperature=0 по умолчанию: воспроизводимость.
    """
    system_message: ChatCompletionSystemMessageParam = {
        "role": "system",
        "content": system_content,
    }
    user_message: ChatCompletionUserMessageParam = {
        "role": "user",
        "content": user_content,
    }
    messages: list[ChatCompletionMessageParam] = [system_message, user_message]
    response = get_client().chat.completions.create(
        model=MODEL,
        temperature=temperature,
        messages=messages,
    )
    content = response.choices[0].message.content
    assert content is not None, "Groq returned a response without text content"
    return content
