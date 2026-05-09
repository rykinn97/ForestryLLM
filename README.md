# ForestryLLM

ForestryLLM 是一个面向林业专业知识问答的可信 RAG 研究项目。当前目标不是先做算法创新型投稿，而是先建立一个可追溯、可复现、能支撑 SCI 应用型论文的知识库与实验流程。

当前已完成第一阶段两本教材语料：

- `林木病理学（第3版）`
- `园林植物遗传育种（第3版）`

注意：`books/` 中的原始书籍材料只用于内部核对，不要公开上传，也不要在论文附件中发布完整原文或大段教材内容。论文中的页码、文献、数据和实验指标都必须能追溯到已核验来源或实际实验输出。

## 新版研究命令行工具

安装依赖：

```bash
pip install -r requirements.txt
```

校验当前知识库：

```bash
python -m forestryllm validate-corpus --output experiments/validation_latest.json
```

运行流程测试评测：

```bash
python -m forestryllm evaluate --qa evaluation/smoke_qa.jsonl --output experiments/latest_evaluation.json
```

把已审核的切块工作表导出为 CSV 和 JSONL：

```bash
python -m forestryllm convert-excel --excel Forestry_KB/chunking_datas/B001_M_林木病理学_切块工作版.xlsx --output-dir Forestry_KB/exports_datas/B001_M_林木病理学
```

导出论文表格指标：

```bash
python -m forestryllm export-paper-tables --input experiments/latest_evaluation.json --output paper/tables/latest_metrics.csv
```

检索单个问题的证据：

```bash
python -m forestryllm retrieve "什么是林木病害？" --method lexical
```

只返回证据和引用、不调用 Ollama 生成答案：

```bash
python -m forestryllm answer "什么是林木病害？" --method lexical --no-generate
```

本地 embedding 环境准备好后，重建 Qdrant 向量索引：

```bash
python -m forestryllm build-index --recreate
```

项目配置文件在：

```text
configs/project_config.json
```

为 `books/` 中尚未入库的书生成 v0.2 草稿切块文件：

```bash
python -m forestryllm prepare-v02-book-drafts
```

完整扩库和人工审校流程见：

```text
docs/v02_knowledge_base_expansion.md
```

## 研究工作区

- `research/`：中文研究日志、决策、盲点和数据治理记录。
- `evaluation/`：问答评测集。`smoke_qa.jsonl` 只用于流程测试，不是正式论文评测集。
- `experiments/`：程序生成的校验和评测结果。
- `paper/`：论文草稿、表格和图。
- `references/`：已核验参考文献记录。

## 旧版原型脚本

这些文件保留用于连续性和对照，但新实验应优先使用 `forestryllm/` 中的 CLI。

- `tool/build_qdrant_from_single_jsonl.py`：从单个 JSONL 建立 Qdrant collection。
- `Forestry_KB/02_add_jsonl_to_qdrant.py`：把一个 JSONL 文件追加写入现有 collection。
- `Forestry_KB/01_excel_to_jsonl_csv.py`：旧版 Excel 转 CSV/JSONL 脚本。
- `tool/check_qdrant_count.py`：检查本地 Qdrant collection 数量。
- `app.py`：早期交互式 RAG 原型。

后续论文实验请优先使用新版 CLI，因为它更容易记录配置、复现结果和追溯实验文件。
