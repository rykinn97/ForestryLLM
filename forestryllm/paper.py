from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .config import resolve_path


def export_metrics_table(result_path: str | Path, output_path: str | Path) -> Path:
    result_resolved = resolve_path(result_path)
    output_resolved = resolve_path(output_path)
    with result_resolved.open("r", encoding="utf-8") as f:
        result: dict[str, Any] = json.load(f)

    output_resolved.parent.mkdir(parents=True, exist_ok=True)
    with output_resolved.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "corpus_version", "retriever", "generator", "top_k", "threshold", "metric", "value"])
        for metric, value in result.get("metrics", {}).items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            writer.writerow([
                result.get("run_id"),
                result.get("corpus_version"),
                result.get("retriever"),
                result.get("generator"),
                result.get("top_k"),
                result.get("threshold"),
                metric,
                value,
            ])
    return output_resolved
