"""Локальная векторная база знаний — без внешних API (экономия токенов).

Документы кодируются векторами в фиксированном пространстве признаков
(hashing-векторизация токенов), поиск — косинусная близость. Индексация
происходит при старте, стоит ноль токенов.

В проде _vectorize заменяется на эмбеддинги через API (OpenAI/Groq) —
интерфейс VectorStore не меняется.
"""
import hashlib
import re
from dataclasses import dataclass

import numpy as np

from data.legal_corpus import LEGAL_CORPUS

_DIM = 4096

_STOPWORDS = {
    "и", "в", "на", "с", "у", "о", "об", "к", "по", "за", "из", "от",
    "а", "но", "или", "что", "как", "это", "то", "же",
    "мне", "меня", "мой", "моя", "мои", "я",
    "есть", "быть", "был", "была",
    "сколько", "какой", "какая", "какие", "когда", "где",
}


def _bucket(token: str) -> int:
    """Стабильный хеш токена (md5 — в отличие от hash(), не меняется между запусками)."""
    return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % _DIM


def _vectorize(text: str, idf: np.ndarray | None = None) -> np.ndarray:
    """Текст → вектор: хешированные слова + символьные n-граммы (3–5).

    Символьные n-граммы ловят морфологию без стемминга: «товар» и «товара»
    дают общие граммы. idf (если передан) приглушает слова, встречающиеся
    во всех документах («товар», «потребитель»).
    """
    vec = np.zeros(_DIM)
    tokens = [t for t in re.findall(r"\w+", text.lower()) if t not in _STOPWORDS]
    for token in tokens:
        vec[_bucket("w:" + token)] += 1.0
    joined = " ".join(tokens)
    for n in (3, 4, 5):
        for i in range(len(joined) - n + 1):
            vec[_bucket(f"g{n}:" + joined[i:i + n])] += 0.3
    if idf is not None:
        vec *= idf
    return vec


@dataclass
class SearchResult:
    doc_id: str
    text: str
    score: float


class VectorStore:
    """Простейший vector store: документы хранятся как векторы, поиск — косинус."""

    def __init__(self, docs: dict[str, str]):
        self._ids = list(docs)
        self._texts = [docs[doc_id] for doc_id in self._ids]
        raw = np.array([_vectorize(t) for t in self._texts])
        # IDF: редкие слова важнее частых («товар» в каждой статье — мало информативен)
        df = np.count_nonzero(raw > 0, axis=0)
        n_docs = len(self._ids)
        self._idf = np.log((1 + n_docs) / (1 + df)) + 1.0
        self._vectors = raw * self._idf
        self._norms = np.linalg.norm(self._vectors, axis=1)

    def search(self, query: str, top_k: int = 2) -> list[SearchResult]:
        """top_k ближайших документов по косинусной близости."""
        q = _vectorize(query, self._idf)
        q_norm = np.linalg.norm(q)
        scores = (self._vectors @ q) / (self._norms * q_norm + 1e-9)
        order = np.argsort(scores)[::-1][:top_k]
        return [
            SearchResult(self._ids[i], self._texts[i], float(scores[i]))
            for i in order
        ]


_store = VectorStore(LEGAL_CORPUS)


def retrieve(question: str, top_k: int = 2) -> list[str]:
    """Вернуть top_k текстов статей, наиболее близких к вопросу."""
    return [r.text for r in _store.search(question, top_k)]


def retrieve_ids(question: str, top_k: int = 2) -> list[str]:
    """Вариант retrieve, возвращающий ID статей — для citation verification."""
    return [r.doc_id for r in _store.search(question, top_k)]
