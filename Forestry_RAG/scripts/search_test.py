from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

qdrant_path = r"/root/autodl-tmp/ForestryLLM/Forestry_RAG/qdrant_data"
collection_name = "forestry_kb"

# 1. 加载模型
model = SentenceTransformer("BAAI/bge-m3")

# 2. 连接本地 Qdrant
client = QdrantClient(path=qdrant_path)

try:
    # 3. 输入测试问题
    query = "松材线虫病主要通过什么传播？"

    # 4. 问题转向量
    query_vector = model.encode(query, normalize_embeddings=True).tolist()

    # 5. 用新接口 query_points
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=5,
        with_payload=True
    )

    # 6. 打印结果
    for i, point in enumerate(results.points, 1):
        payload = point.payload
        print(f"\n=== 结果 {i} ===")
        print("score:", point.score)
        print("chunk_id:", payload.get("chunk_id"))
        print("book_title:", payload.get("book_title"))
        print("chapter_title:", payload.get("chapter_title"))
        print("section_title:", payload.get("section_title"))
        print("chunk_title:", payload.get("chunk_title"))
        print("citation_anchor:", payload.get("citation_anchor"))
        print("cleaned_text:", payload.get("cleaned_text"))

finally:
    client.close()