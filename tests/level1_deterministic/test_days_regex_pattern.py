"""Уровень 1: параметризация regex в чистом виде.

Не зовёт Groq. Проверяет, что шаблон `\\b14\\b` совпадает
только с изолированным числом 14 и не ловит "140" или "2014".
"""
import re
import pytest

pytestmark = pytest.mark.deterministic


@pytest.mark.parametrize("text, should_match", [
    ("14 дней", True),
    ("140 дней", False),
    ("в 2014 году", False),
    ("Итог: 14", True),
    ("Строка: 14.", True),
    ("Строка с числом 114", False),
])
def test_days_regex_matches_only_isolated_14(text, should_match):
    assert should_match == bool(re.search(r"\b14\b", text)), \
        f"regex не сработал как ожидалось на строке {text!r} (ждали {should_match})"

