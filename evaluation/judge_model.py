"""Инфраструктура уровня 3: судья-модель для DeepEval-метрик.

Оборачивает Groq в интерфейс DeepEvalBaseLLM + instructor в JSON-режиме
(tool-calling ломается на openai/gpt-oss-20b).
Используется только LLM-judge тестами: test_answer_relevancy, test_hallucination.
"""
import os
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionUserMessageParam
import instructor
from deepeval.models.base_model import DeepEvalBaseLLM
from pydantic import BaseModel

load_dotenv()

class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, model="openai/gpt-oss-20b"):
        self.model = model
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to the project .env file.")
        base_client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        self.client = instructor.from_openai(base_client, mode=instructor.Mode.JSON)

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        user_message: ChatCompletionUserMessageParam = {"role": "user", "content": prompt}
        messages: list[ChatCompletionMessageParam] = [user_message]
        # reasoning_effort="low": судья отвечает коротким JSON — глубокие
        # рассуждения только жгут токены (важно на бесплатном тарифе Groq).
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_model=schema,
            reasoning_effort="low",
        )

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.model
