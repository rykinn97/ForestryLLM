import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ========= 1. 路径设置 =========
jsonl_path = Path(r"/root/autodl-tmp/ForestryLLM/Foresty_KB/03_exports/Forestry_KB_merged_total.jsonl")
qdrant_path = r"/root/autodl-tmp/ForestryLLM/Forestry_RAG/qdrant_data"
collection_name = "forestry_kb"

# ========= 2. 读取 JSONL =========
records = []
with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

print(f"读取到 {len(records)} 条知识块")

# ========= 3. 加载 embedding 模型 =========
print("正在加载 embedding 模型...")
model = SentenceTransformer("BAAI/bge-m3")

# ========= 4. 取出要做 embedding 的文本 =========
texts = [r["cleaned_text"] for r in records]

print("正在生成向量，请稍等...")
embeddings = model.encode(
    texts,
    batch_size=16,
    show_progress_bar=True,
    normalize_embeddings=True
)

# 向量维度
vector_size = len(embeddings[0])
print(f"向量维度: {vector_size}")

# ========= 5. 连接 Qdrant 本地库 =========
# 这里用 path=...，表示本地磁盘模式，不需要单独启动服务器
client = QdrantClient(path=qdrant_path)

# ========= 6. 创建 collection =========
# 如果已存在同名 collection，可以先删掉再重建
existing = [c.name for c in client.get_collections().collections]
if collection_name in existing:
    client.delete_collection(collection_name=collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=vector_size,
        distance=Distance.COSINE
    )
)

print(f"已创建 collection: {collection_name}")

# ========= 7. 组织 points =========
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

# ========= 8. 批量写入 =========
client.upsert(
    collection_name=collection_name,
    points=points
)

print(f"成功写入 {len(points)} 条数据到 Qdrant")
print("完成")