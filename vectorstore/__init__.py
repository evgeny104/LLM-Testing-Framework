"""Пакет векторной базы знаний: данные (knowledge_base/), загрузчик, поиск."""
from vectorstore.corpus import LEGAL_CORPUS, get_article
from vectorstore.store import VectorStore, retrieve, retrieve_ids

__all__ = [
    "LEGAL_CORPUS",
    "VectorStore",
    "get_article",
    "retrieve",
    "retrieve_ids",
]
