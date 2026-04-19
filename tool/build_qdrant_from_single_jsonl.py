import json
import os
from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# =========================
# 0. 离线设置
# =========================
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# =========================
# 1. 路径配置
# =========================
# 改成你当前已有的 林木病理学 JSONL 文件路径
jsonl_path = Path("/root/autodl-tmp/ForestryLLM/Forestry_KB/exports_datas/B001_M_林木病理学/B001_M_林木病理学_chunks_clean.jsonl")

# 以后所有脚本都统一用这个 Qdrant 目录
qdrant_path = "/root/autodl-tmp/ForestryLLM/Forestry_KB/qdrant_data"

collection_name = "forestry_kb"

# 本地离线 bge-m3 路径
bge_model_path = "/root/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"

# =========================
# 2. 读取 JSONL
# =========================
records = []
with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

print(f"读取到知识块 {len(records)} 条")

if len(records) == 0:
    raise ValueError("JSONL 为空，没有可建库的数据。")

# =========================
# 3. 加载 embedding 模型
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

vector_size = len(embeddings[0])
print(f"向量维度: {vector_size}")

# =========================
# 5. 连接 Qdrant
# =========================
print("正在连接 Qdrant...")
client = QdrantClient(path=qdrant_path)

# =========================
# 6. 如果已有同名 collection，就先删掉重建
# =========================
collections = client.get_collections().collections
collection_names = [c.name for c in collections]

if collection_name in collection_names:
    print(f"检测到已有 collection: {collection_name}，先删除再重建")
    client.delete_collection(collection_name=collection_name)

# =========================
# 7. 创建 collection
# =========================
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=vector_size,
        distance=Distance.COSINE
    )
)

print(f"已创建 collection: {collection_name}")

# =========================
# 8. 组织 points
# =========================
points = []
for idx, (record, vector) in enumerate(zip(records, embeddings)):
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
        id=idx,
        vector=vector.tolist(),
        payload=payload
    )
    points.append(point)

# =========================
# 9. 全量写入
# =========================
print("正在写入 Qdrant...")
client.upsert(
    collection_name=collection_name,
    points=points
)

print(f"成功写入 {len(points)} 条数据到 collection: {collection_name}")
client.close()