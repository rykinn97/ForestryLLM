# 评测集说明

`smoke_qa.jsonl` 只用于回归测试和流程验证，不代表正式论文评测集。

正式评测集要求：

- 每条样本必须绑定 `gold_chunk_ids` 或明确标记为 `unanswerable`。
- `review_status` 必须从 `draft` 升级到 `reviewed` 后才能进入论文主实验。
- `expected_answer_notes` 只写证据范围或审核提示，不编造标准答案。

字段：

- `question`
- `answerability`: `answerable` 或 `unanswerable`
- `gold_chunk_ids`
- `gold_citation_anchors`
- `expected_answer_notes`
- `reviewer`
- `review_status`

