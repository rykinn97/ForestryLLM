# 数据治理规范

## 语料边界

第一阶段固定使用：

- `Forestry_KB/exports_datas/B001_M_林木病理学/B001_M_林木病理学_chunks_clean.jsonl`
- `Forestry_KB/exports_datas/B002_M_园林植物遗传育种/B002_M_园林植物遗传育种_chunks_clean.jsonl`

`books/` 目录中的原始书籍文件仅用于内部核对，不作为公开数据发布。

## 知识块字段

继续使用现有 schema：`chunk_id`, `book_title`, `chapter_title`, `section_title`, `page_start`, `page_end`, `chunk_title`, `topic_type`, `cleaned_text`, `keywords`, `citation_anchor`。

## 审核规则

- 页码、章节、引用锚点必须能回到原始资料核验。
- `cleaned_text` 必须保持原意，不允许补写原文没有的结论。
- 所有正式评测样本必须标明审核状态，未审核样本不能用于论文主结论。

