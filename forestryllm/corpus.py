from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import resolve_path


REQUIRED_FIELDS = [
    "chunk_id",
    "book_title",
    "chapter_title",
    "section_title",
    "page_start",
    "page_end",
    "chunk_title",
    "topic_type",
    "cleaned_text",
    "keywords",
    "citation_anchor",
]

DEFAULT_COLUMN_MAPPING = {
    "知识块编号": "chunk_id",
    "书名": "book_title",
    "章标题": "chapter_title",
    "节标题": "section_title",
    "起始页码": "page_start",
    "结束页码": "page_end",
    "知识块标题": "chunk_title",
    "内容类型": "topic_type",
    "清洗后正文": "cleaned_text",
    "关键词": "keywords",
    "引用锚点": "citation_anchor",
}


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    summary: dict[str, Any]
    errors: list[dict[str, Any]]
    warnings: list[dict[str, Any]]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    resolved = resolve_path(path)
    rows: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {resolved}:{line_number}: {exc}") from exc
            row["_source_file"] = str(resolved)
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def load_corpus(corpus_files: Iterable[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for corpus_file in corpus_files:
        records.extend(read_jsonl(corpus_file))
    return records


def convert_excel_to_csv_jsonl(excel_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    import json as json_module

    import pandas as pd

    input_path = resolve_path(excel_path)
    resolved_output_dir = resolve_path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(input_path)
    df = df.rename(columns=DEFAULT_COLUMN_MAPPING)
    missing = [field for field in REQUIRED_FIELDS if field not in df.columns]
    if missing:
        raise ValueError(f"Excel missing required columns: {missing}")

    df = df[REQUIRED_FIELDS].copy()
    df = df.dropna(how="all")
    df = df[df["cleaned_text"].notna()]
    for col in [
        "chunk_id",
        "book_title",
        "chapter_title",
        "section_title",
        "chunk_title",
        "topic_type",
        "cleaned_text",
        "keywords",
        "citation_anchor",
    ]:
        df[col] = df[col].astype(str).str.strip()
    df["topic_type"] = df["topic_type"].str.lower()
    df["page_start"] = pd.to_numeric(df["page_start"], errors="coerce").astype("Int64")
    df["page_end"] = pd.to_numeric(df["page_end"], errors="coerce").astype("Int64")
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["chunk_id"], keep="first")

    base_name = input_path.stem.replace("_切块工作版", "").replace(" ", "_")
    csv_path = resolved_output_dir / f"{base_name}_chunks_clean.csv"
    jsonl_path = resolved_output_dir / f"{base_name}_chunks_clean.jsonl"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in df.to_dict(orient="records"):
            f.write(json_module.dumps(record, ensure_ascii=False) + "\n")

    return {
        "input": str(input_path),
        "csv_path": str(csv_path),
        "jsonl_path": str(jsonl_path),
        "records_exported": len(df),
        "duplicates_removed": before_dedup - len(df),
    }


def validate_records(records: list[dict[str, Any]], allowed_topic_types: Iterable[str]) -> ValidationResult:
    allowed = set(allowed_topic_types)
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}
    book_counts: Counter[str] = Counter()
    topic_counts: Counter[str] = Counter()
    file_counts: Counter[str] = Counter()
    duplicate_ids: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    for idx, record in enumerate(records):
        location = {
            "source_file": record.get("_source_file"),
            "line_number": record.get("_line_number"),
            "record_index": idx,
            "chunk_id": record.get("chunk_id"),
        }

        missing = [field for field in REQUIRED_FIELDS if record.get(field) in (None, "")]
        if missing:
            errors.append({**location, "type": "missing_required_fields", "fields": missing})

        chunk_id = record.get("chunk_id")
        if chunk_id:
            if chunk_id in seen:
                duplicate_ids[chunk_id].append(location)
            else:
                seen[chunk_id] = location

        topic_type = record.get("topic_type")
        if topic_type:
            topic_counts[str(topic_type)] += 1
            if topic_type not in allowed:
                errors.append({**location, "type": "invalid_topic_type", "value": topic_type})

        page_start = _safe_int(record.get("page_start"))
        page_end = _safe_int(record.get("page_end"))
        if page_start is None:
            errors.append({**location, "type": "invalid_page_start", "value": record.get("page_start")})
        if page_end is None:
            errors.append({**location, "type": "invalid_page_end", "value": record.get("page_end")})
        if page_start is not None and page_end is not None and page_end < page_start:
            errors.append({**location, "type": "page_end_before_page_start", "page_start": page_start, "page_end": page_end})

        cleaned_text = str(record.get("cleaned_text") or "").strip()
        if cleaned_text and len(cleaned_text) < 20:
            warnings.append({**location, "type": "very_short_cleaned_text", "length": len(cleaned_text)})

        book_counts[str(record.get("book_title") or "UNKNOWN")] += 1
        file_counts[str(record.get("_source_file") or "UNKNOWN")] += 1

    for chunk_id, locations in duplicate_ids.items():
        first = seen.get(chunk_id)
        errors.append({
            "type": "duplicate_chunk_id",
            "chunk_id": chunk_id,
            "first": first,
            "duplicates": locations,
        })

    summary = {
        "total_records": len(records),
        "book_counts": dict(sorted(book_counts.items())),
        "topic_counts": dict(sorted(topic_counts.items())),
        "file_counts": dict(sorted(file_counts.items())),
        "duplicate_chunk_id_count": len(duplicate_ids),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "required_fields": REQUIRED_FIELDS,
    }
    return ValidationResult(ok=not errors, summary=summary, errors=errors, warnings=warnings)


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
