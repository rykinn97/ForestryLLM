from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def lexical_search(question: str, records: list[dict[str, Any]], top_k: int = 8) -> list[dict[str, Any]]:
    query_terms = _terms(question)
    if not query_terms:
        return []
    query_counter = Counter(query_terms)
    scored: list[dict[str, Any]] = []

    for record in records:
        text = " ".join(
            str(record.get(field) or "")
            for field in ["chunk_title", "keywords", "chapter_title", "section_title", "cleaned_text"]
        )
        doc_terms = _terms(text)
        if not doc_terms:
            continue
        score = _cosine(query_counter, Counter(doc_terms))
        if score <= 0:
            continue
        scored.append({
            "score": score,
            "chunk_id": record.get("chunk_id"),
            "book_title": record.get("book_title"),
            "chapter_title": record.get("chapter_title"),
            "section_title": record.get("section_title"),
            "chunk_title": record.get("chunk_title"),
            "cleaned_text": record.get("cleaned_text"),
            "citation_anchor": record.get("citation_anchor"),
            "topic_type": record.get("topic_type"),
        })

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def _terms(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _cosine(a: Counter[str], b: Counter[str]) -> float:
    overlap = set(a) & set(b)
    numerator = sum(a[key] * b[key] for key in overlap)
    if numerator == 0:
        return 0.0
    a_norm = math.sqrt(sum(value * value for value in a.values()))
    b_norm = math.sqrt(sum(value * value for value in b.values()))
    return numerator / (a_norm * b_norm)
