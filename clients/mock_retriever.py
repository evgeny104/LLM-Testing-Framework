"""
Что моделирует: настоящий retriever с vector store и эмбеддингами.
Почему keyword-match: детерминизм в тестах + достаточно "туповат", чтобы
иногда выдавать нерелевантные статьи (полезно для тестов context precision).
В проде заменяется на vector-search по эмбеддингам.
"""
import re
from data.legal_corpus import LEGAL_CORPUS

_STOPWORDS = {
      "и", "в", "на", "с", "у", "о", "об", "к", "по", "за", "из", "от",
      "а", "но", "или", "что", "как", "это", "то", "же",
      "мне", "меня", "мой", "моя", "мои", "я",
      "есть", "быть", "был", "была",
      "сколько", "какой", "какая", "какие", "когда", "где",
  }
def _tokenize(text: str) -> set[str]:
    """Разбить текст на набор значимых слов (lowercase, без стоп-слов и пунктуации)."""
    words = re.findall(r"\w+", text.lower())
    return {w for w in words if w not in _STOPWORDS}


def retrieve(question: str, top_k: int = 2) -> list[str]:
    """Вернуть top_k текстов статей, наиболее релевантных вопросу.
    Ранжирование: по числу общих слов между вопросом и текстом статьи.
    Возвращает тексты (list[str]) — формат под retrieval_context в RAGAS/DeepEval.
    """
    q_tokens = _tokenize(question)
    scored = [
        (text, len(q_tokens & _tokenize(text)))
        for text in LEGAL_CORPUS.values()
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [text for text, _ in scored[:top_k]]


def retrieve_ids(question: str, top_k: int = 2) -> list[str]:
    """Вариант retrieve, возвращающий ID статей — для citation verification."""
    q_tokens = _tokenize(question)
    scored = [
        (aid, len(q_tokens & _tokenize(text)))
        for aid, text in LEGAL_CORPUS.items()
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [aid for aid, _ in scored[:top_k]]
