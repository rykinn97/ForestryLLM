from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import resolve_path
from .corpus import REQUIRED_FIELDS, validate_records


DEFAULT_EXISTING_CORPUS_FILES = [
    "Forestry_KB/exports_datas/B001_M_林木病理学/B001_M_林木病理学_chunks_clean.jsonl",
    "Forestry_KB/exports_datas/B002_M_园林植物遗传育种/B002_M_园林植物遗传育种_chunks_clean.jsonl",
]

BOOKS_TO_ADD = [
    {
        "book_id": "B003",
        "short_name": "园林树木",
        "title": "园林树木",
        "source_hint": "园林树木 (何会流 主编)",
        "chunk_prefix": "M_YLSM_V1",
    },
    {
        "book_id": "B004",
        "short_name": "园林树木学",
        "title": "园林树木学",
        "source_hint": "园林树木学",
        "chunk_prefix": "M_YLSMX_V1",
    },
    {
        "book_id": "B005",
        "short_name": "林木化学保护学",
        "title": "林木化学保护学",
        "source_hint": "林木化学保护学",
        "chunk_prefix": "M_LMHXBHX_V1",
    },
    {
        "book_id": "B006",
        "short_name": "树木学南方本",
        "title": "树木学 南方本",
        "source_hint": "树木学 南方本",
        "chunk_prefix": "M_SMXNFB_V1",
    },
    {
        "book_id": "B007",
        "short_name": "森林培育学",
        "title": "森林培育学",
        "source_hint": "森林培育学",
        "chunk_prefix": "M_SLPYX_V1",
    },
    {
        "book_id": "B008",
        "short_name": "森林昆虫学原理",
        "title": "森林昆虫学原理（第5版）",
        "source_hint": "森林昆虫学原理",
        "chunk_prefix": "M_SLKCXYL_V1",
    },
    {
        "book_id": "B009",
        "short_name": "森林生态学",
        "title": "森林生态学",
        "source_hint": "森林生态学",
        "chunk_prefix": "M_SLSTX_V1",
    },
    {
        "book_id": "B010",
        "short_name": "经济林栽培学",
        "title": "经济林栽培学（第4版）",
        "source_hint": "经济林栽培学 第4版",
        "chunk_prefix": "M_JJLZPX_V1",
    },
]

TOPIC_TYPES = [
    "definition",
    "pathogen",
    "symptom",
    "occurrence_rule",
    "transmission",
    "diagnosis",
    "control_method",
    "summary",
]

CHINESE_COLUMNS = {
    "chunk_id": "知识块编号",
    "book_title": "书名",
    "chapter_title": "章标题",
    "section_title": "节标题",
    "page_start": "起始页码",
    "page_end": "结束页码",
    "chunk_title": "知识块标题",
    "topic_type": "内容类型",
    "cleaned_text": "清洗后正文",
    "keywords": "关键词",
    "citation_anchor": "引用锚点",
    "review_status": "审校状态",
    "source_file": "来源文件",
    "ingestion_note": "入库说明",
}


@dataclass(frozen=True)
class PageText:
    page: int
    text: str


def prepare_v02_book_drafts(
    books_dir: str | Path = "books",
    chunking_dir: str | Path = "Forestry_KB/chunking_datas",
    exports_dir: str | Path = "Forestry_KB/exports_datas",
    report_path: str | Path = "experiments/v02_book_ingestion_report.json",
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    books_root = resolve_path(books_dir)
    chunking_root = resolve_path(chunking_dir)
    exports_root = resolve_path(exports_dir)
    report_resolved = resolve_path(report_path)
    chunking_root.mkdir(parents=True, exist_ok=True)
    exports_root.mkdir(parents=True, exist_ok=True)
    report_resolved.parent.mkdir(parents=True, exist_ok=True)

    outputs = []
    all_new_records: list[dict[str, Any]] = []
    for spec in BOOKS_TO_ADD:
        source = _find_source_file(books_root, spec["source_hint"])
        pages = _extract_pages(source)
        records = _records_from_pages(spec, source, pages)
        all_new_records.extend(records)

        workbook_path = chunking_root / f'{spec["book_id"]}_{spec["short_name"]}_切块工作版.xlsx'
        export_dir = exports_root / f'{spec["book_id"]}_{spec["short_name"]}'
        export_dir.mkdir(parents=True, exist_ok=True)
        csv_path = export_dir / f'{spec["book_id"]}_{spec["short_name"]}_chunks_clean.csv'
        jsonl_path = export_dir / f'{spec["book_id"]}_{spec["short_name"]}_chunks_clean.jsonl'

        _write_workbook(workbook_path, records)
        _write_csv(csv_path, records)
        _write_jsonl(jsonl_path, records)

        validation = validate_records(records, TOPIC_TYPES)
        outputs.append({
            "book_id": spec["book_id"],
            "book_title": spec["title"],
            "source_file": str(source),
            "source_extension": source.suffix.lower(),
            "pages_extracted": len(pages),
            "chunks_exported": len(records),
            "workbook": str(workbook_path),
            "csv": str(csv_path),
            "jsonl": str(jsonl_path),
            "validation_ok": validation.ok,
            "validation_summary": validation.summary,
            "validation_errors": validation.errors[:20],
            "review_status": "draft_auto_extracted_needs_human_review",
        })

    corpus_files = DEFAULT_EXISTING_CORPUS_FILES + [
        _project_relative_path(Path(book["jsonl"])) for book in outputs
    ]
    config_output = None
    if config_path is not None:
        config_output = _write_v02_config(config_path, corpus_files)

    combined_validation = validate_records(all_new_records, TOPIC_TYPES)
    report = {
        "corpus_version_target": "v0.2-expanded-corpus",
        "status": "draft_auto_extracted_needs_human_review",
        "note": (
            "These files use the standard knowledge-base schema and are suitable for "
            "human review. They should be treated as draft until chunk titles, pages, "
            "keywords, topic types, and OCR/text-extraction quality are checked."
        ),
        "books": outputs,
        "corpus_files": corpus_files,
        "config_output": config_output,
        "new_record_count": len(all_new_records),
        "combined_validation_ok": combined_validation.ok,
        "combined_validation_summary": combined_validation.summary,
        "combined_validation_errors": combined_validation.errors[:50],
    }
    with report_resolved.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    report["report_path"] = str(report_resolved)
    return report


def _find_source_file(books_root: Path, source_hint: str) -> Path:
    candidates = sorted(path for path in books_root.iterdir() if path.is_file() and source_hint in path.name)
    if not candidates:
        raise FileNotFoundError(f"No book source under {books_root} matched {source_hint!r}")
    return candidates[0]


def _extract_pages(source: Path) -> list[PageText]:
    suffix = source.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf_pages(source)
    if suffix == ".epub":
        return _extract_epub_pages(source)
    raise ValueError(f"Unsupported book format: {source}")


def _extract_pdf_pages(source: Path) -> list[PageText]:
    import fitz

    pages: list[PageText] = []
    with fitz.open(source) as doc:
        for page_index, page in enumerate(doc, 1):
            text = _clean_text(page.get_text("text"))
            if text:
                pages.append(PageText(page=page_index, text=text))
    return pages


def _extract_epub_pages(source: Path) -> list[PageText]:
    from bs4 import BeautifulSoup
    from ebooklib import ITEM_DOCUMENT
    from ebooklib import epub

    book = epub.read_epub(str(source))
    pages: list[PageText] = []
    page_number = 1
    for item in book.get_items_of_type(ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "lxml")
        text = _clean_text(soup.get_text("\n"))
        if not text:
            continue
        pages.append(PageText(page=page_number, text=text))
        page_number += 1
    return pages


def _records_from_pages(spec: dict[str, str], source: Path, pages: Iterable[PageText]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    chapter_title = "未识别章节"
    section_title = "未识别小节"

    for page in pages:
        for paragraph in _split_page_text(page.text):
            heading = _heading_from_text(paragraph)
            if heading:
                if _looks_like_chapter(heading):
                    chapter_title = heading
                    section_title = heading
                else:
                    section_title = heading
                continue
            if len(paragraph) < 80:
                continue
            for chunk_text in _split_long_paragraph(paragraph):
                chunk_number = len(records) + 1
                chunk_id = f'{spec["chunk_prefix"]}_{chunk_number:04d}'
                chunk_title = _make_chunk_title(chunk_text)
                topic_type = _infer_topic_type(chunk_text)
                keywords = _make_keywords(chunk_text, chunk_title)
                citation_anchor = (
                    f'{spec["title"]}-{chapter_title}-{section_title}-{page.page}-{page.page}'
                )
                records.append({
                    "chunk_id": chunk_id,
                    "book_title": spec["title"],
                    "chapter_title": chapter_title,
                    "section_title": section_title,
                    "page_start": page.page,
                    "page_end": page.page,
                    "chunk_title": chunk_title,
                    "topic_type": topic_type,
                    "cleaned_text": chunk_text,
                    "keywords": keywords,
                    "citation_anchor": citation_anchor,
                    "review_status": "draft_auto_extracted_needs_human_review",
                    "source_file": source.name,
                    "ingestion_note": "自动抽取草稿；入正式论文实验前需人工审校。",
                })
    return records


def _split_page_text(text: str) -> list[str]:
    parts = re.split(r"\n{2,}|(?<=[。！？；])\s+", text)
    return [_clean_text(part) for part in parts if _clean_text(part)]


def _split_long_paragraph(text: str, max_chars: int = 850) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[。！？；])", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if current and len(current) + len(sentence) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current += sentence
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if len(chunk) >= 80]


def _heading_from_text(text: str) -> str | None:
    normalized = text.strip(" .·•\t")
    if len(normalized) > 40:
        return None
    patterns = [
        r"^第[一二三四五六七八九十百0-9]+[章节篇编].*",
        r"^[0-9]+(\.[0-9]+){0,3}\s*\S+.*",
        r"^[一二三四五六七八九十]+[、.]\s*\S+.*",
    ]
    if any(re.match(pattern, normalized) for pattern in patterns):
        return normalized
    return None


def _looks_like_chapter(text: str) -> bool:
    return bool(re.match(r"^第[一二三四五六七八九十百0-9]+[章节篇编]", text))


def _make_chunk_title(text: str) -> str:
    first_sentence = re.split(r"[。！？；]", text, maxsplit=1)[0]
    title = re.sub(r"\s+", "", first_sentence)
    title = re.sub(r"^[（(]?[0-9一二三四五六七八九十]+[）)、.．]\s*", "", title)
    if len(title) > 42:
        title = title[:42]
    return title or "自动抽取知识块"


def _infer_topic_type(text: str) -> str:
    rules = [
        ("control_method", ["防治", "控制", "治理", "药剂", "施药", "栽培措施", "管理措施"]),
        ("diagnosis", ["诊断", "鉴别", "识别", "检疫", "测定", "调查方法"]),
        ("symptom", ["症状", "病状", "危害状", "表现为", "叶斑", "枯萎", "腐烂"]),
        ("pathogen", ["病原", "真菌", "细菌", "病毒", "线虫", "昆虫", "害虫", "病菌"]),
        ("transmission", ["传播", "传染", "扩散", "媒介", "侵染循环"]),
        ("occurrence_rule", ["发生规律", "流行", "影响因素", "生态", "分布", "适生", "生境"]),
        ("definition", ["是指", "称为", "定义", "概念", "所谓"]),
    ]
    for topic, keywords in rules:
        if any(keyword in text for keyword in keywords):
            return topic
    return "summary"


def _make_keywords(text: str, title: str) -> str:
    candidates = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,12}", title + "。" + text)
    stopwords = {
        "以及",
        "因此",
        "由于",
        "可以",
        "进行",
        "具有",
        "主要",
        "一般",
        "不同",
        "影响",
        "包括",
        "形成",
        "发生",
        "植物",
    }
    seen: list[str] = []
    for candidate in candidates:
        if candidate in stopwords or candidate.isdigit():
            continue
        if candidate not in seen:
            seen.append(candidate)
        if len(seen) >= 8:
            break
    return ";".join(seen[:8]) if seen else title


def _clean_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _write_workbook(path: Path, records: list[dict[str, Any]]) -> None:
    df = _records_dataframe(records)
    df.to_excel(path, index=False)


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    df = _records_dataframe(records)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            row = {field: record[field] for field in REQUIRED_FIELDS}
            for extra in ["review_status", "source_file", "ingestion_note"]:
                row[extra] = record[extra]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _records_dataframe(records: list[dict[str, Any]]):
    import pandas as pd

    columns = REQUIRED_FIELDS + ["review_status", "source_file", "ingestion_note"]
    rows = [{CHINESE_COLUMNS[column]: record[column] for column in columns} for record in records]
    return pd.DataFrame(rows, columns=[CHINESE_COLUMNS[column] for column in columns])


def _write_v02_config(config_path: str | Path, corpus_files: list[str]) -> str:
    resolved = resolve_path(config_path)
    with resolved.open("r", encoding="utf-8") as f:
        config = json.load(f)
    config["corpus_version"] = "v0.2-expanded-corpus"
    config["corpus_files"] = corpus_files
    with resolved.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return str(resolved)


def _project_relative_path(path: Path) -> str:
    root = resolve_path(".")
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()
