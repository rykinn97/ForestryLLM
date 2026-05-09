# ForestryLLM

ForestryLLM is being organized as a long-running trusted RAG research project for forestry question answering. The first paper track is an SCI-oriented applied study, not an algorithm-first CCF submission.

The current stage uses two internally held textbook-derived corpora only:

- `林木病理学（第3版）`
- `园林植物遗传育种（第3版）`

Do not publish the original `books/` materials or large verbatim textbook content. Paper claims, page numbers, literature, data, and experiment results must be traceable to verified sources or actual experiment outputs.

## New Research CLI

Install dependencies:

```bash
pip install -r requirements.txt
```

Validate the frozen corpus:

```bash
python -m forestryllm validate-corpus --output experiments/validation_latest.json
```

Run the smoke evaluation:

```bash
python -m forestryllm evaluate --qa evaluation/smoke_qa.jsonl --output experiments/latest_evaluation.json
```

Convert an audited chunking workbook:

```bash
python -m forestryllm convert-excel --excel Forestry_KB/chunking_datas/B001_M_林木病理学_切块工作版.xlsx --output-dir Forestry_KB/exports_datas/B001_M_林木病理学
```

Export metrics for the manuscript:

```bash
python -m forestryllm export-paper-tables --input experiments/latest_evaluation.json --output paper/tables/latest_metrics.csv
```

Retrieve evidence for one question:

```bash
python -m forestryllm retrieve "什么是林木病害？" --method lexical
```

Answer without calling Ollama, useful for checking citations:

```bash
python -m forestryllm answer "什么是林木病害？" --method lexical --no-generate
```

Build or rebuild a Qdrant index when the local embedding environment is ready:

```bash
python -m forestryllm build-index --recreate
```

Configuration lives in `configs/project_config.json`.

## Research Workspace

- `research/`: Chinese research logs, decisions, blind spots, and data governance.
- `evaluation/`: QA evaluation sets. `smoke_qa.jsonl` is only a workflow test and is not a formal paper benchmark.
- `experiments/`: generated validation and evaluation outputs.
- `paper/`: English manuscript scaffold, tables, and figures.
- `references/`: verified bibliography notes.

## Legacy Prototype Scripts

These files are preserved for continuity and comparison:

- `tool/build_qdrant_from_single_jsonl.py`: build a Qdrant collection from one JSONL file.
- `Forestry_KB/02_add_jsonl_to_qdrant.py`: add one JSONL file to an existing Qdrant collection.
- `Forestry_KB/01_excel_to_jsonl_csv.py`: convert a chunking spreadsheet to CSV/JSONL.
- `tool/check_qdrant_count.py`: check the local Qdrant collection count.
- `app.py`: original interactive RAG prototype.

The new CLI should be preferred for reproducible paper experiments.
