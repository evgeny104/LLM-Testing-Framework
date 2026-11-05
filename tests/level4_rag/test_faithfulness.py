"""
Уровень 4 — RAGAS Faithfulness. Проверяет, что ответ модели опирается только на retrieval_context,
а не на внутренние знания LLM.
"""

import pytest
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness

from clients.groq_client import get_client
from clients.mock_retriever import retrieve
from groq.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

load_dotenv()

pytestmark = [
    pytest.mark.rag,
    pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY is not set"),
]


groq_llm = ChatOpenAI(
    model="openai/gpt-oss-20b",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY", "missing-api-key"),
    temperature=0,
    max_tokens=4096,
)

ragas_llm = LangchainLLMWrapper(groq_llm)


def test_rag_wrapper_builds():
    assert ragas_llm is not None


def test_faithfulness_minimal():
    question = "Сколько дней у меня есть на возврат товара?"
    contexts = retrieve(question, top_k=2)

    system_message: ChatCompletionSystemMessageParam = {
        "role": "system",
        "content": "Отвечай только на основе этих статей закона:\n\n" + "\n\n".join(contexts),
    }
    user_message: ChatCompletionUserMessageParam = {"role": "user", "content": question}
    messages: list[ChatCompletionMessageParam] = [system_message, user_message]
    sut_response = get_client().chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        temperature=0,
    )
    answer = sut_response.choices[0].message.content
    assert answer is not None, "Groq returned a response without text content"

    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
    )
    dataset = EvaluationDataset(samples=[sample])

    result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(llm=ragas_llm)],
    )

    score = result.to_pandas()["faithfulness"].iloc[0]
    assert score >= 0.5, f"Faithfulness score {score} below threshold 0.5"

