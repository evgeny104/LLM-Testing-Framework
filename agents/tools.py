"""Инструменты агента: схема для модели, реестр, вызов в режиме tool-calling.

Контракт агента: модель решает, какой инструмент вызвать и с какими
аргументами; исполняет вызов приложение (см. реестр TOOLS).
"""
from llm.groq_client import MODEL, get_client
from vectorstore import retrieve

# Объявление инструмента для модели: имя, описание, схема аргументов.
RETRIEVE_TOOL = {
    "type": "function",
    "function": {
        "name": "retrieve",
        "description": "Поиск релевантных статей закона по базе знаний",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Поисковый запрос"},
            },
            "required": ["query"],
        },
    },
}

# Реестр инструментов агента: имя → исполняемая функция.
TOOLS = {
    "retrieve": retrieve,
}

# Реальный паттерн RAG-агентов: system-промпт ОБЯЗЫВАЕТ искать в базе.
# Без этого модель отвечает из памяти обучения (проверено: игнорирует
# инструмент при tool_choice="auto").
AGENT_SYSTEM_PROMPT = (
    "Ты юридический ассистент. Перед ответом на любой вопрос о законе "
    "обязательно ищи в базе знаний через инструмент retrieve. Отвечай "
    "только на основе найденных статей."
)


def ask_agent(messages, tools=(RETRIEVE_TOOL,)):
    """Один шаг диалога с агентом в режиме tool-calling."""
    return get_client().chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=list(tools),
        tool_choice="auto",
        temperature=0,
    )
