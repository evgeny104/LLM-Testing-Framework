"""Клиент Groq и инфраструктура вызовов модели.

Groq играет две роли в проекте:
- SUT (complete) — тестируемая модель: текстовые ответы по контексту;
- инфраструктура оценки (complete_structured) — instructor-клиент для
  структурированного вывода (уровень 2) и ответов судьи (уровень 3).

JSON-режим instructor используем, т.к. tool-calling ломается на
openai/gpt-oss-20b.
"""
import os
from functools import lru_cache
from typing import TypeVar

from dotenv import load_dotenv
from groq import Groq
from groq.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from instructor import Mode, from_openai
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam as OpenAIMessageParam,
    ChatCompletionSystemMessageParam as OpenAISystemMessageParam,
    ChatCompletionUserMessageParam as OpenAIUserMessageParam,
)
from pydantic import BaseModel

load_dotenv()

MODEL = "openai/gpt-oss-20b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

T = TypeVar("T", bound=BaseModel)


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


@lru_cache(maxsize=1)
def get_instructor_client():
    """instructor-клиент поверх OpenAI-совместимого эндпоинта Groq (JSON-режим)."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to the project .env file.")
    return from_openai(
        OpenAI(api_key=api_key, base_url=GROQ_BASE_URL),
        mode=Mode.JSON,
    )


def complete_structured(
    response_model: type[T],
    user_content: str,
    system_content: str | None = None,
    model: str = MODEL,
) -> T:
    """Структурированный вызов: модель отвечает JSON строго по pydantic-схеме.

    instructor парсит ответ и валидирует схему; при ошибке повторяет запрос
    с подсказкой. reasoning_effort="low": ответы короткие — экономим токены.
    """
    messages: list[OpenAIMessageParam] = []
    if system_content is not None:
        system_message: OpenAISystemMessageParam = {
            "role": "system",
            "content": system_content,
        }
        messages.append(system_message)
    user_message: OpenAIUserMessageParam = {"role": "user", "content": user_content}
    messages.append(user_message)
    return get_instructor_client().chat.completions.create(
        model=model,
        messages=messages,
        response_model=response_model,
        reasoning_effort="low",
    )
