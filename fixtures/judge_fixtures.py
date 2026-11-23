"""Фикстуры модели-судьи DeepEval (уровень 3)."""
import pytest
from judge.judge_model import GroqJudge


@pytest.fixture(scope="module")
def judge():
    """Модель-судья DeepEval (GroqJudge) — общая на все тесты уровня 3."""
    return GroqJudge()
