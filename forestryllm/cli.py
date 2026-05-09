from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .book_ingestion import prepare_v02_book_drafts
from .config import DEFAULT_CONFIG, load_config
from .corpus import convert_excel_to_csv_jsonl, load_corpus, validate_records
from .evaluation import evaluate_retrieval, read_qa, write_json
from .paper import export_metrics_table
from .rag import INSUFFICIENT_EVIDENCE_ANSWER, build_citations, build_messages, call_ollama
from .retrieval import lexical_search


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="forestryllm", description="ForestryLLM trusted RAG research CLI")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to project config JSON")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-corpus", help="Validate corpus schema and quality gates")
    validate_parser.add_argument("--output", help="Optional JSON output path")

    convert_parser = subparsers.add_parser("convert-excel", help="Convert a chunking Excel workbook to CSV and JSONL")
    convert_parser.add_argument("--excel", required=True, help="Input Excel workbook")
    convert_parser.add_argument("--output-dir", required=True, help="Output directory for CSV/JSONL")
    convert_parser.add_argument("--report", help="Optional JSON report path")

    build_parser = subparsers.add_parser("build-index", help="Build a Qdrant index from configured corpus")
    build_parser.add_argument("--recreate", action="store_true", help="Delete and recreate the configured collection")
    build_parser.add_argument("--output", help="Optional JSON output path")

    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve evidence for one question")
    retrieve_parser.add_argument("question")
    retrieve_parser.add_argument("--method", choices=["lexical", "qdrant"], default="lexical")
    retrieve_parser.add_argument("--top-k", type=int)
    retrieve_parser.add_argument("--output")

    answer_parser = subparsers.add_parser("answer", help="Answer one question using retrieved evidence")
    answer_parser.add_argument("question")
    answer_parser.add_argument("--method", choices=["lexical", "qdrant"], default="lexical")
    answer_parser.add_argument("--top-k", type=int)
    answer_parser.add_argument("--threshold", type=float)
    answer_parser.add_argument("--no-generate", action="store_true", help="Return evidence/citations without calling Ollama")
    answer_parser.add_argument("--output")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate retrieval and rejection behavior on QA JSONL")
    eval_parser.add_argument("--qa", default="evaluation/smoke_qa.jsonl")
    eval_parser.add_argument("--top-k", type=int)
    eval_parser.add_argument("--threshold", type=float)
    eval_parser.add_argument("--output", default="experiments/latest_evaluation.json")

    export_parser = subparsers.add_parser("export-paper-tables", help="Export paper-ready metrics CSV from an evaluation JSON")
    export_parser.add_argument("--input", default="experiments/latest_evaluation.json")
    export_parser.add_argument("--output", default="paper/tables/latest_metrics.csv")

    v02_parser = subparsers.add_parser("prepare-v02-book-drafts", help="Prepare draft chunks for books not yet in the corpus")
    v02_parser.add_argument("--books-dir", default="books")
    v02_parser.add_argument("--chunking-dir", default="Forestry_KB/chunking_datas")
    v02_parser.add_argument("--exports-dir", default="Forestry_KB/exports_datas")
    v02_parser.add_argument("--report", default="experiments/v02_book_ingestion_report.json")
    v02_parser.add_argument(
        "--update-config",
        default=None,
        help="Optional config JSON to update to v0.2 after draft files are generated, for example configs/project_config.json",
    )

    args = parser.parse_args(argv)
    config = load_config(args.config)

    if args.command == "validate-corpus":
        records = load_corpus(config["corpus_files"])
        result = validate_records(records, config["allowed_topic_types"])
        payload = {"ok": result.ok, "summary": result.summary, "errors": result.errors, "warnings": result.warnings}
        _emit(payload, args.output)
        if not result.ok:
            raise SystemExit(1)
        return

    if args.command == "convert-excel":
        payload = convert_excel_to_csv_jsonl(args.excel, args.output_dir)
        _emit(payload, args.report)
        return

    if args.command == "build-index":
        from .index import build_index

        records = load_corpus(config["corpus_files"])
        result = validate_records(records, config["allowed_topic_types"])
        if not result.ok:
            raise SystemExit("Corpus validation failed; run validate-corpus for details.")
        payload = build_index(config, records, recreate=args.recreate)
        _emit(payload, args.output)
        return

    if args.command == "retrieve":
        records = load_corpus(config["corpus_files"]) if args.method == "lexical" else []
        top_k = args.top_k or config["retrieval"]["top_k"]
        retrieved = _retrieve(config, records, args.question, args.method, top_k)
        _emit({"question": args.question, "method": args.method, "retrieved": retrieved}, args.output)
        return

    if args.command == "answer":
        records = load_corpus(config["corpus_files"]) if args.method == "lexical" else []
        top_k = args.top_k or config["retrieval"]["top_k"]
        threshold = args.threshold if args.threshold is not None else _default_threshold(config, args.method)
        retrieved = _retrieve(config, records, args.question, args.method, top_k)
        evidence = [item for item in retrieved if item["score"] >= threshold][: config["retrieval"]["max_context_chunks"]]
        if not evidence:
            payload = {"question": args.question, "answer": INSUFFICIENT_EVIDENCE_ANSWER, "citations": [], "evidence": retrieved}
        elif args.no_generate:
            payload = {"question": args.question, "answer": None, "citations": build_citations(evidence), "evidence": evidence}
        else:
            answer = call_ollama(config, build_messages(args.question, evidence))
            payload = {"question": args.question, "answer": answer, "citations": build_citations(evidence), "evidence": evidence}
        _emit(payload, args.output)
        return

    if args.command == "evaluate":
        records = load_corpus(config["corpus_files"])
        qa_rows = read_qa(args.qa)
        top_k = args.top_k or config["retrieval"]["top_k"]
        threshold = args.threshold if args.threshold is not None else config["retrieval"].get("lexical_smoke_threshold", config["retrieval"]["threshold"])
        payload = evaluate_retrieval(qa_rows, records, config, top_k, threshold)
        _emit(payload, args.output)
        return

    if args.command == "export-paper-tables":
        output = export_metrics_table(args.input, args.output)
        _emit({"output": str(output)}, None)
        return

    if args.command == "prepare-v02-book-drafts":
        payload = prepare_v02_book_drafts(
            args.books_dir,
            args.chunking_dir,
            args.exports_dir,
            args.report,
            config_path=args.update_config,
        )
        _emit(payload, None)
        return


def _retrieve(config: dict[str, Any], records: list[dict[str, Any]], question: str, method: str, top_k: int) -> list[dict[str, Any]]:
    if method == "qdrant":
        from .index import qdrant_search

        return qdrant_search(config, question, top_k)
    return lexical_search(question, records, top_k)


def _default_threshold(config: dict[str, Any], method: str) -> float:
    if method == "lexical":
        return config["retrieval"].get("lexical_smoke_threshold", 0.0)
    return config["retrieval"]["threshold"]


def _emit(payload: dict[str, Any], output: str | None) -> None:
    if output:
        path = write_json(output, payload)
        print(json.dumps({"output": str(path)}, ensure_ascii=False, indent=2))
        return
    print(json.dumps(payload, ensure_ascii=False, indent=2))
