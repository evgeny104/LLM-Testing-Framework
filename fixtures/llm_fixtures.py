"""Переиспользуемые pytest-фикстуры для LLM-тестов.

Реализация — здесь, регистрация — через conftest.py уровней
(см. tests/level*/conftest.py). Фикстуры построены фабриками:
тестовый файл вызывает фабрику на уровне модуля и получает фикстуру
со своим именем и параметрами.
"""
import pytest
from clients.groq_client import complete

QUOTE_DIRECTIVE = (
    "В ответе обязательно указывай номер статьи в формате 'Статья {article}'."
)


def make_model_answer(cases):
    """Фабрика параметризованной фикстуры: один вызов Groq на кейс
    (scope="module" — результат кэшируется и раздаётся всем тестам файла).

    Каждый кейс: dict с id, question, context, expected_article, expected_days.
    Фикстура возвращает {**case, "answer": <текст ответа модели>}.
    """
    @pytest.fixture(scope="module", params=cases, ids=[c["id"] for c in cases])
    def _model_answer(request):
        case = request.param
        answer = complete(
            system_content=(
                f"Отвечай только на основе этого текста закона: {case['context']}. "
                + QUOTE_DIRECTIVE.format(article=case["expected_article"])
            ),
            user_content=case["question"],
        )
        return {**case, "answer": answer}

    return _model_answer


def make_single_answer(system_content, question):
    """Фабрика простой фикстуры: один вызов Groq на файл, возвращает строку-ответ."""
    @pytest.fixture(scope="module")
    def _model_answer():
        return complete(system_content=system_content, user_content=question)

    return _model_answer
