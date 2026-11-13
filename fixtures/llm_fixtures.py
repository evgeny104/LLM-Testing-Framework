"""Переиспользуемые pytest-фикстуры для LLM-тестов.

Реализация — здесь, регистрация — через conftest.py уровней
(см. tests/level*/conftest.py). Фикстуры построены фабриками:
тестовый файл вызывает фабрику на уровне модуля и получает фикстуру
со своим именем и параметрами.
"""
import re
import pytest
from clients.groq_client import complete
from data.legal_corpus import get_article

# Номер статьи извлекаем из текста корпуса — эталон и контекст из одного источника.
ARTICLE_NUMBER = re.compile(r"Статья (\d+)")
QUOTE_DIRECTIVE = (
    "В ответе обязательно указывай номер статьи в формате 'Статья {article}'."
)


def make_model_answer(cases):
    """Фабрика параметризованной фикстуры: один вызов Groq на кейс
    (scope="module" — результат кэшируется и раздаётся всем тестам файла).

    Каждый кейс: dict с id, question, article_id, expected_days
    (см. golden_cases.py). Контекст достаётся из базы знаний по article_id,
    номер статьи — из текста этого контекста.

    Фикстура возвращает {**case, "context", "expected_article", "answer"}.
    """
    @pytest.fixture(scope="module", params=cases, ids=[c["id"] for c in cases])
    def _model_answer(request):
        case = request.param
        context = get_article(case["article_id"])
        article = int(ARTICLE_NUMBER.search(context).group(1))
        answer = complete(
            system_content=(
                f"Отвечай только на основе этого текста закона: {context}. "
                + QUOTE_DIRECTIVE.format(article=article)
            ),
            user_content=case["question"],
        )
        return {**case, "context": context, "expected_article": article, "answer": answer}

    return _model_answer


def make_single_answer(system_content, question):
    """Фабрика простой фикстуры: один вызов Groq на файл, возвращает строку-ответ."""
    @pytest.fixture(scope="module")
    def _model_answer():
        return complete(system_content=system_content, user_content=question)

    return _model_answer
