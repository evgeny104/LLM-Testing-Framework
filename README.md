# LLM-Testing-Framework

![tests](https://img.shields.io/badge/tests-pytest-blue)
![llm](https://img.shields.io/badge/LLM-Groq-orange)

Фреймворк для тестирования и оценки LLM-приложений. Демонстрирует
практическую методологию проверки ответов языковых моделей: от
детерминированных regex-проверок до оценки «моделью-судьёй»
(LLM-as-a-Judge) и метрик RAG-пайплайнов.

Тестируемая система (SUT) — LLM через Groq API на примере юридического
ассистента (закон о защите прав потребителей). Проект демонстрирует навыки
QA-инженера в области AI: тест-дизайн для недетерминированных систем,
эталонные наборы данных, изоляция зависимостей, воспроизводимость прогонов.

## Возможности

- **Детерминированные проверки** (уровень 1): regex-assertions против
  эталонных значений в mini golden dataset; параметризованные фикстуры
  экономят вызовы API.
- **LLM-as-a-Judge** (уровень 3): метрики релевантности ответа
  и галлюцинаций через DeepEval с кастомной judge-моделью (Groq + instructor).
- **RAG-оценка** (уровень 4): проверка обоснованности ответа (Faithfulness)
  через RAGAS на синтетическом корпусе юридических статей.
- **Изоляция и воспроизводимость**: маркеры разделяют офлайн- и API-тесты,
  `temperature=0`, детерминированный mock-retriever вместо vector store.

## Стек

Python 3 · pytest · Groq API · DeepEval · RAGAS · instructor · langchain-openai

## Быстрый старт

```bash
# 1. Клонировать и создать виртуальное окружение
git clone <repo-url> && cd LLM-Testing-Framework
python -m venv .venv && source .venv/Scripts/activate   # Win: .venv\Scripts\activate

# 2. Установить зависимости
pip install -r requirements.txt

# 3. Добавить API-ключ Groq
cp .env.example .env        # Windows: copy .env.example .env
# в .env вписать свой ключ: GROQ_API_KEY=gsk_...

# 4. Запустить тесты
pytest -v                   # всё сразу
pytest -m "not llm_api" -v  # только офлайн-тесты (без API-ключа)
pytest -m llm_judge -v      # только оценки LLM-as-a-Judge
```

## Структура проекта

```
clients/                  # Тестируемая система и её зависимости
├── groq_client.py        #   клиент к Groq API (SUT — модель под тестом)
├── legal_corpus.py       #   синтетический корпус юридических статей
└── mock_retriever.py     #   keyword-retriever (имитация vector store)

evaluation/
└── judge_model.py        # обёртка Groq в интерфейс DeepEval (оценщик)

tests/
├── level1_deterministic/ # regex-проверки по эталонам, golden dataset
├── level3_llm_judge/     # DeepEval: релевантность и галлюцинации
└── level4_rag/           # RAGAS: Faithfulness на retrieval-контексте

scripts/
└── list_models_grok.py   # утилита: список моделей Groq → output.json
```

## Уровни тестирования

| Уровень | Что проверяет | Инструменты | API-ключ |
|---------|---------------|-------------|----------|
| 1 | Факты: цитата статьи, число дней, непустота | pytest, regex, эталоны | да¹ |
| 3 | Семантика: релевантность, галлюцинации | DeepEval + judge-модель | да |
| 4 | RAG: опирается ли ответ на контекст | RAGAS (Faithfulness) | да |

¹ Кроме `test_days_regex_pattern.py` — чистый regex, работает офлайн.

Ключевая идея уровня 1: один вызов модели (фикстура `scope="module"`)
покрывается несколькими независимыми проверками — эталонные значения заданы
в данных (CASES), а не в коде тестов.

## Маркеры pytest

Маркеры заданы в `pytest.ini`, позволяют выборочно запускать тесты:

| Маркер | Назначение |
|--------|-----------|
| `deterministic` | офлайн-проверки, не требуют API |
| `llm_api` | вызывают Groq, без ключа пропускаются |
| `llm_judge` | оценки DeepEval, расходуют токены |
| `rag` | проверки RAGAS, расходуют токены |

## Дорожная карта

- [ ] Уровень 2: валидация структурированного вывода (JSON-схемы)
- [ ] Устойчивость: перефразирование, prompt injection, jailbreak
- [ ] Расширение RAG-метрик: context precision/recall, answer correctness
- [ ] Стабильность метрик: повторные прогоны, допуски (threshold tuning)

## Полезные команды

```bash
# чистый regex, офлайн, без API-ключа
pytest tests/level1_deterministic/test_days_regex_pattern.py -v

# список доступных моделей Groq → output.json
python scripts/list_models_grok.py
```
