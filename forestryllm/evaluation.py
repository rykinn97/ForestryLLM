from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import resolve_path
from .retrieval import lexical_search


def read_qa(path: str | Path) -> list[dict[str, Any]]:
    resolved = resolve_path(path)
    rows: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_source_file"] = str(resolved)
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def evaluate_retrieval(
    qa_rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    config: dict[str, Any],
    top_k: int,
    threshold: float,
) -> dict[str, Any]:
    cases = []
    answerable_cases = 0
    hit_at_k = 0
    reciprocal_ranks = []
    ndcgs = []
    unanswerable_cases = 0
    rejection_correct = 0

    for row in qa_rows:
        question = row["question"]
        answerability = row.get("answerability", "answerable")
        gold_ids = set(row.get("gold_chunk_ids") or [])
        retrieved = lexical_search(question, records, top_k=top_k)
        retrieved_ids = [item.get("chunk_id") for item in retrieved]
        top_score = retrieved[0]["score"] if retrieved else 0.0

        case = {
            "question": question,
            "answerability": answerability,
            "gold_chunk_ids": list(gold_ids),
            "retrieved": [
                {
                    "rank": idx + 1,
                    "score": item["score"],
                    "chunk_id": item.get("chunk_id"),
                    "citation_anchor": item.get("citation_anchor"),
                    "chunk_title": item.get("chunk_title"),
                }
                for idx, item in enumerate(retrieved)
            ],
        }

        if answerability == "unanswerable":
            unanswerable_cases += 1
            rejected = top_score < threshold
            if rejected:
                rejection_correct += 1
            case["rejected_by_threshold"] = rejected
        else:
            answerable_cases += 1
            first_rank = _first_relevant_rank(retrieved_ids, gold_ids)
            if first_rank is not None:
                hit_at_k += 1
                reciprocal_ranks.append(1.0 / first_rank)
            else:
                reciprocal_ranks.append(0.0)
            ndcgs.append(_binary_ndcg(retrieved_ids, gold_ids, top_k))
            case["first_relevant_rank"] = first_rank

        cases.append(case)

    metrics = {
        "answerable_cases": answerable_cases,
        "unanswerable_cases": unanswerable_cases,
        f"recall_at_{top_k}": hit_at_k / answerable_cases if answerable_cases else None,
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else None,
        "ndcg": sum(ndcgs) / len(ndcgs) if ndcgs else None,
        "rejection_accuracy": rejection_correct / unanswerable_cases if unanswerable_cases else None,
        "citation_hit_rate": hit_at_k / answerable_cases if answerable_cases else None,
        "answer_support_rate": None,
        "unsupported_generation_rate": None,
        "notes": "answer_support_rate and unsupported_generation_rate require human or LLM-judge answer audits and are intentionally not inferred here.",
    }

    return {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config.get("_config_path"),
        "corpus_version": config.get("corpus_version"),
        "retriever": "lexical_cosine_smoke",
        "generator": None,
        "top_k": top_k,
        "threshold": threshold,
        "metrics": metrics,
        "cases": cases,
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return resolved


def _first_relevant_rank(retrieved_ids: list[str | None], gold_ids: set[str]) -> int | None:
    for idx, chunk_id in enumerate(retrieved_ids, 1):
        if chunk_id in gold_ids:
            return idx
    return None


def _binary_ndcg(retrieved_ids: list[str | None], gold_ids: set[str], top_k: int) -> float:
    if not gold_ids:
        return 0.0
    dcg = 0.0
    for idx, chunk_id in enumerate(retrieved_ids[:top_k], 1):
        if chunk_id in gold_ids:
            dcg += 1.0 / math.log2(idx + 1)
    ideal_hits = min(len(gold_ids), top_k)
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0
