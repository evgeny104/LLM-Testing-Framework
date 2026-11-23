"""База знаний: тексты статей загружаются из knowledge_base/*.txt.

Единый источник текстов для всех уровней: golden dataset (ссылки по ID),
RAG (векторная база), judge-метрики. ID статьи = имя файла без расширения.
Добавить статью в базу — положить txt-файл в knowledge_base/, код не меняется.
"""
from pathlib import Path

KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parent / "knowledge_base"


def _load() -> dict[str, str]:
    return {
        path.stem: path.read_text(encoding="utf-8").strip()
        for path in sorted(KNOWLEDGE_BASE_DIR.glob("*.txt"))
    }


LEGAL_CORPUS: dict[str, str] = _load()


def get_article(article_id: str) -> str:
    """Достать текст статьи по ID. Кидает KeyError, если ID нет в корпусе."""
    return LEGAL_CORPUS[article_id]
