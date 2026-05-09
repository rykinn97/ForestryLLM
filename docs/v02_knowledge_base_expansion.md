# ForestryLLM v0.2 知识库扩充说明

本文档记录如何把当前两本已审核书籍的知识库，扩充为 v0.2 草稿知识库。

## 目标

为 `books/` 中尚未进入知识库的 8 本书生成可人工审校的切块工作表，并同步生成标准 CSV/JSONL 导出文件。

这些文件只是自动抽取草稿，不能直接当成正式论文实验语料。正式使用前必须人工核对章节、页码、切块边界、关键词、主题标签和正文抽取质量。

## 本轮新增书籍

- `B003_园林树木`
- `B004_园林树木学`
- `B005_林木化学保护学`
- `B006_树木学南方本`
- `B007_森林培育学`
- `B008_森林昆虫学原理`
- `B009_森林生态学`
- `B010_经济林栽培学`

已有的 `B001_林木病理学` 和 `B002_园林植物遗传育种` 不会被 v0.2 草稿生成命令重复处理。

## 安装依赖

生成草稿前，请你自己运行：

```powershell
pip install -r requirements.txt
```

新增的文本抽取依赖包括：

- `PyMuPDF`：用于 PDF 文本抽取。
- `ebooklib`、`beautifulsoup4`、`lxml`：用于 EPUB 文本抽取。

## 生成草稿切块

运行：

```powershell
python -m forestryllm prepare-v02-book-drafts
```

预期会生成：

- `Forestry_KB/chunking_datas/B003_园林树木_切块工作版.xlsx`
- `Forestry_KB/chunking_datas/B004_园林树木学_切块工作版.xlsx`
- `Forestry_KB/chunking_datas/B005_林木化学保护学_切块工作版.xlsx`
- `Forestry_KB/chunking_datas/B006_树木学南方本_切块工作版.xlsx`
- `Forestry_KB/chunking_datas/B007_森林培育学_切块工作版.xlsx`
- `Forestry_KB/chunking_datas/B008_森林昆虫学原理_切块工作版.xlsx`
- `Forestry_KB/chunking_datas/B009_森林生态学_切块工作版.xlsx`
- `Forestry_KB/chunking_datas/B010_经济林栽培学_切块工作版.xlsx`
- `Forestry_KB/exports_datas/` 下对应的 `*_chunks_clean.csv` 和 `*_chunks_clean.jsonl`
- `experiments/v02_book_ingestion_report.json`

## 更新配置

确认草稿文件已经生成后，如果要把主配置切换到 v0.2，运行：

```powershell
python -m forestryllm prepare-v02-book-drafts --update-config configs/project_config.json
```

该命令会把：

- `corpus_version` 改为 `v0.2-expanded-corpus`
- `corpus_files` 改为 B001/B002 加 B003-B010 的 10 本书 JSONL 路径

如果暂时不想覆盖主配置，可以先用模板配置校验：

```powershell
python -m forestryllm --config configs/project_config.v0.2.template.json validate-corpus --output experiments/validation_v02_expanded_corpus.json
```

## 校验语料

切换到 v0.2 后运行：

```powershell
python -m forestryllm validate-corpus --output experiments/validation_v02_expanded_corpus.json
```

使用 v0.2 做正式实验前，必须先查看校验报告。

## 人工审校要求

自动生成的行会带有：

```text
review_status=draft_auto_extracted_needs_human_review
```

冻结 v0.2 语料前，必须人工检查：

- OCR 或文本抽取质量。
- 章标题和节标题是否正确。
- 页码是否能回到原书核验。
- 切块边界是否保持一个知识块一个核心主题。
- `chunk_title` 是否准确。
- `topic_type` 是否符合项目统一标签。
- `keywords` 是否能辅助检索。
- `citation_anchor` 是否可追溯。

只有完成这些审校后，才能把该语料作为正式的 `v0.2-expanded-corpus`。
