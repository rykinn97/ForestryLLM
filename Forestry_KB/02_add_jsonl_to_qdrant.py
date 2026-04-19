import json
import os
from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# =========================
# 0. 离线设置
# =========================
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# =========================
# 1. 路径配置
# =========================
new_jsonl_path = Path("/root/autodl-tmp/ForestryLLM/Forestry_KB/exports_datas/B002_M_园林植物遗传育种/B002_M_园林植物遗传育种_chunks_clean.jsonl")  # 改成你的 JSONL 路径

qdrant_path = "/root/autodl-tmp/ForestryLLM/Forestry_KB/qdrant_data"   # 改成你的 Qdrant 本地库路径
collection_name = "forestry_kb"

bge_model_path = "/root/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"

# =========================
# 2. 读取新增 JSONL
# =========================
records = []
with open(new_jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

print(f"读取到新增知识块 {len(records)} 条")

if len(records) == 0:
    raise ValueError("JSONL 为空，没有可入库的数据。")

# =========================
# 3. 加载本地 embedding 模型
# =========================
print("正在加载 embedding 模型...")
model = SentenceTransformer(
    bge_model_path,
    device="cpu",
    local_files_only=True
)

# =========================
# 4. 生成向量
# =========================
texts = [r["cleaned_text"] for r in records]

print("正在生成向量...")
embeddings = model.encode(
    texts,
    batch_size=16,
    show_progress_bar=True,
    normalize_embeddings=True
)

# =========================
# 5. 连接 Qdrant
# =========================
print("正在连接 Qdrant...")
client = QdrantClient(path=qdrant_path)

# =========================
# 6. 获取当前已有数据量，给新增数据续号
# =========================
count_result = client.count(collection_name=collection_name)
start_id = count_result.count
print(f"当前 collection 已有 {start_id} 条数据")

# =========================
# 7. 组织 points
# =========================
points = []
for i, (record, vector) in enumerate(zip(records, embeddings)):
    payload = {
        "chunk_id": record.get("chunk_id"),
        "book_title": record.get("book_title"),
        "chapter_title": record.get("chapter_title"),
        "section_title": record.get("section_title"),
        "page_start": record.get("page_start"),
        "page_end": record.get("page_end"),
        "chunk_title": record.get("chunk_title"),
        "topic_type": record.get("topic_type"),
        "cleaned_text": record.get("cleaned_text"),
        "keywords": record.get("keywords"),
        "citation_anchor": record.get("citation_anchor"),
    }

    point = PointStruct(
        id=start_id + i,
        vector=vector.tolist(),
        payload=payload
    )
    points.append(point)

# =========================
# 8. 增量写入 Qdrant
# =========================
print("正在写入 Qdrant...")
client.upsert(
    collection_name=collection_name,
    points=points
)

print(f"成功新增写入 {len(points)} 条数据到 collection: {collection_name}")

client.close()