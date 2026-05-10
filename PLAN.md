# ForestryLLM v0.2 知识库扩充计划

**Implementation Changes**
- 保留已入库两本：`林木病理学 第3版`、`园林植物遗传育种（第3版）`，不重复处理。
- 新增 1 本书，建议编号：`B003_M_园林树木学`
- 每本书先生成切块工作表到 `Forestry_KB/chunking_datas/`，字段严格沿用现有标准：`chunk_id、book_title、chapter_title、section_title、page_start、page_end、chunk_title、topic_type、cleaned_text、keywords、citation_anchor`。
- 切块通过人工审校后，再导出到 `Forestry_KB/exports_datas/<book_id>_M_<book_name>/` 下的 `*_chunks_clean.csv` 和 `*_chunks_clean.jsonl`。
- 更新 `configs/project_config.json`：
  `corpus_version` 改为 `v0.2-expanded-corpus`，`corpus_files` 加入 10 本书的 JSONL。
- 重新构建 Qdrant 索引，但不把这一步作为检索器对比实验，只作为 v0.2 知识库准备。

**Processing Rules**

- 不直接把 JSON 全文粗暴切入知识库；必须保留章节、页码、引用锚点和主题标签。
- PDF/EPUB 自动抽取只能作为草稿来源，最终入库前要人工审校标题、页码、正文清洗、关键词和 topic_type。
- 主题标签先沿用现有 8 类；如新增书中出现明显不适配内容，再统一扩展标签，不逐书随意加标签。
- 优先处理与论文“林业可信 RAG”覆盖最相关的书：森林培育、林木化学保护、森林昆虫、森林生态；园林树木/树木学/经济林用于扩大植物与栽培覆盖。

**Test Plan**
- 每新增一本书后运行 corpus validation，检查必填字段、重复 `chunk_id`、页码、topic_type。
- 每本书入库后生成新的 validation 报告，记录总 chunk 数、各书 chunk 数、topic_type 分布。
- 对现有 B001/B002 QA 做一次回归检索，确认扩库没有明显破坏旧题召回。

**Assumptions**
- 本轮目标是“扩库并冻结 v0.2”，不是提升检索指标。
- `books/` 中两本已入库书只用于核对，不重复生成新 chunk。
- 如果需要安装依赖，安装在环境forest中，python版本为3.10
