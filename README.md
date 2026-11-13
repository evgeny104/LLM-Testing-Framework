# LLM-Testing-Framework

![tests](https://img.shields.io/badge/tests-pytest-blue)
![llm](https://img.shields.io/badge/LLM-Groq-orange)

Фреймворк для тестирования LLM-приложений на примере юридического ассистента
(закон о защите прав потребителей, Groq API). Тестируется **один продукт**, а
уровни — это разные контракты качества его ответа: от regex-эталонов до
оценки судьёй-моделью и метрик RAG.

Схемы устройства — поток теста и RAG-пайплайн: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Уровни тестирования

| Уровень | Контракт качества | Как проверяет | Статус |
|---------|-------------------|---------------|--------|
| 1 | Факты: модель назвала статью и число дней из контекста | regex по эталонам (golden dataset) | ✅ |
| 2 | Формат: ответ — JSON строго по схеме | instructor + pydantic, сверка полей | ✅ |
| 3 | Смысл: ответ по делу и без выдумок | DeepEval-метрики, судья GroqJudge | ✅ |
| 4 | RAG: ответ опирается на найденный контекст | RAGAS Faithfulness | ✅ |
| 5 | Стабильность: повторяемость при t=0, допуск при t>0 | повторные прогоны | 🚧 |
| 6 | Устойчивость: перефразирование, prompt injection, jailbreak | — | ⬜ |
| 7 | Агенты: вызовы Skills/Tools, интеграции через MCP | — | ⬜ |

Принцип: один и тот же вопрос пропускается через уровни, как багаж через
разные виды досмотра. Упавший уровень сразу показывает класс дефекта:
сломался формат — уровень 2, ответ невпопад — уровень 3, плавает от прогона
к прогону — уровень 5.

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
pytest -m stability -v      # только уровень 5
```

## Структура проекта

```
clients/                  # дверь к модели и поиск
├── groq_client.py        #   complete() и complete_structured() — все вызовы Groq
└── mock_retriever.py     #   keyword-retriever (имитация vector store)

data/
└── legal_corpus.py       # база знаний: тексты статей, единый источник

evaluation/
└── judge_model.py        # судья DeepEval (обёртка Groq для метрик)

fixtures/                 # pytest-фикстуры; регистрация — в conftest.py
├── llm_fixtures.py       #   фабрики фикстур вызова модели
├── judge_fixtures.py     #   судья DeepEval
└── rag_fixtures.py       #   LLM-обёртка RAGAS

tests/
├── level1_deterministic/ # regex по эталонам
│   ├── golden_cases.py   #   golden dataset: кейсы со ссылками на корпус
│   ├── test_factual_dataset.py
│   ├── test_factual_checks.py
│   └── test_days_regex_pattern.py  # чистый regex, офлайн
├── level2_structured/    # валидация JSON по pydantic-схеме
├── level3_llm_judge/     # DeepEval: релевантность и галлюцинации
├── level4_rag/           # RAGAS Faithfulness на retrieval-контексте
├── level5_stability/     # повторные прогоны, допуски
├── level6_robustness/    # перефразирование, prompt injection, jailbreak
└── level7_agents/        # Skills/Tools, интеграции API и MCP

scripts/
└── list_models_grok.py   # утилита: список моделей Groq → output.json
```

## Маркеры pytest

Маркеры заданы в `pytest.ini`, позволяют выборочно запускать тесты:

| Маркер | Назначение |
|--------|-----------|
| `deterministic` | офлайн-проверки, не требуют API |
| `llm_api` | вызывают Groq, без ключа пропускаются |
| `llm_judge` | оценки DeepEval, расходуют токены |
| `rag` | проверки RAGAS, расходуют токены |
| `stability` | повторные прогоны, анализ стабильности |
| `structured` | валидация структурированного вывода |

## Дорожная карта

Реализовано: уровни 1–4, первые тесты уровня 5. Дальше по плану:

- [ ] **Уровень 5 (продолжение):** статистика по большему числу прогонов,
      подбор обоснованных допусков.
- [ ] **Уровень 6:** устойчивость к перефразированию вопроса; проверка защиты
      от Prompt Injection и Jailbreak.
- [ ] **Уровень 4 (расширение):** RAGAS context precision/recall,
      answer correctness.
- [ ] **Уровень 7:** тестирование AI-агентов — вызовы Skills/Tools,
      интеграции через API и MCP.
- [ ] **CI:** GitHub Actions — офлайн-тесты на каждый пуш.

## Полезные команды


```bash
# чистый regex, офлайн, без API-ключа
pytest tests/level1_deterministic/test_days_regex_pattern.py -v

# список доступных моделей Groq → output.json
python scripts/list_models_grok.py
```
