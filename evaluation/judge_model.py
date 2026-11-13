"""Инфраструктура уровня 3: судья-модель для DeepEval-метрик.

Реализует интерфейс DeepEvalBaseLLM: DeepEval зовёт .generate(prompt, schema).
Структурированный вызов делегирует clients.groq_client.complete_structured
(instructor в JSON-режиме — tool-calling ломается на openai/gpt-oss-20b).
"""
from pydantic import BaseModel
from deepeval.models.base_model import DeepEvalBaseLLM

from clients.groq_client import complete_structured


class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, model="openai/gpt-oss-20b"):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return complete_structured(schema, user_content=prompt, model=self.model)

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.model
