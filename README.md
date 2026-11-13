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

## Как работает фреймворк

```
┌────────────┐  1. просит данные  ┌─────────────┐  2. complete()  ┌────────────┐
│   тест     │ ─────────────────► │  фикстура   │ ──────────────► │  Groq API  │
│ (проверка) │                    │ (подготовка)│   HTTP-запрос   │   (SUT)    │
└─────┬──────┘                    └──────┬──────┘                 └────────────┘
      │                                  │ 3. ответ (текст)
      │ 4. получает                      ▼
      │                            ┌─────────────┐
      │                            │    ответ    │ ◄── ур. 4: контекст
      │                            └──────┬──────┘     из mock_retriever
      ▼                                   │
5. assert: ответ правильный? ◄────────────┘
```

Что проверяет assert на каждом уровне:

- **Уровень 1** — regex по эталонам из golden dataset (`golden_cases.py`).
- **Уровень 2** — объект `LegalAnswer`, валидированный pydantic-схемой.
- **Уровень 3** — DeepEval-метрики; судья — `evaluation/judge_model.py`.
- **Уровень 4** — RAGAS Faithfulness: опирается ли ответ на найденный контекст.

Общий вызов модели — `complete()` в `clients/groq_client.py`: одна точка,
к которой обращаются все фикстуры (temperature=0, воспроизводимость).

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
└── mock_retriever.py     #   keyword-retriever (имитация vector store)

data/
└── legal_corpus.py       # база знаний: синтетический корпус юридических статей

evaluation/
└── judge_model.py        # обёртка Groq в интерфейс DeepEval (оценщик)

fixtures/                 # реализации фикстур; регистрация — в conftest.py
├── llm_fixtures.py       #   фабрики фикстур вызова Groq (параметризованные)
└── judge_fixtures.py     #   judge-модель DeepEval и LLM-обёртка RAGAS

tests/
├── level1_deterministic/ # regex-проверки по эталонам, golden dataset
├── level2_structured/    # валидация JSON-вывода по pydantic-схемам
├── level3_llm_judge/     # DeepEval: релевантность и галлюцинации
├── level4_rag/           # RAGAS: Faithfulness на retrieval-контексте
├── level5_stability/     # повторные прогоны, стабильность метрик
├── level6_robustness/    # перефразирование, prompt injection, jailbreak
└── level7_agents/        # Skills/Tools, интеграции API и MCP

scripts/
└── list_models_grok.py   # утилита: список моделей Groq → output.json
```

Уровни 2, 5–7 — в разработке: структура и маркеры готовы, тесты добавляются
по плану из раздела «Дорожная карта».

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
| `stability` | повторные прогоны, анализ стабильности |
| `robustness` | перефразирование, injection, jailbreak |
| `agents` | тесты агентов: Skills/Tools, MCP |

## Дорожная карта

Каждый пункт привязан к инженерной задаче тестирования LLM:

- [x] **Автотесты для LLM на Python/pytest** — уровень 1: фикстуры,
      параметризация, эталоны, маркеры.
- [ ] **Валидация JSON-структур по схемам** — уровень 2: instructor +
      pydantic, модель отвечает строго типизированным объектом.
- [x] **Релевантность и галлюцинации (DeepEval)** — уровень 3:
      AnswerRelevancyMetric, HallucinationMetric.
- [x] **Оценка через LLM-as-a-Judge** — уровень 3: кастомный judge
      (GroqJudge) в интерфейсе DeepEvalBaseLLM.
- [ ] **RAGAS: полнота контекста и точность ответов** — уровень 4:
      добавить context precision/recall и answer correctness.
- [ ] **Недетерминированность и стабильность** — уровень 5: повторные
      прогоны, разброс ответов, допуски (threshold tuning).
- [ ] **Устойчивость к перефразированию** — уровень 6: эталон сохраняется
      при перефразированном вопросе.
- [ ] **Защита от Prompt Injection и Jailbreak** — уровень 6: модель не
      нарушает системный промпт под атакующими вставками.
- [ ] **AI-агенты: Skills/Tools, MCP** — уровень 7: тестирование вызовов
      инструментов и интеграций с внешними сервисами.

## Полезные команды

```bash
# чистый regex, офлайн, без API-ключа
pytest tests/level1_deterministic/test_days_regex_pattern.py -v

# список доступных моделей Groq → output.json
python scripts/list_models_grok.py
```
