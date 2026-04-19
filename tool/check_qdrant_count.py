from qdrant_client import QdrantClient

qdrant_path = "/root/autodl-tmp/ForestryLLM/Forestry_KB/qdrant_data"   # 改成你的实际路径
collection_name = "forestry_kb"

client = QdrantClient(path=qdrant_path)

collections = [c.name for c in client.get_collections().collections]
print("当前 collections:", collections)

if collection_name not in collections:
    print(f"未找到 collection: {collection_name}")
else:
    count_result = client.count(collection_name=collection_name)
    print(f"{collection_name} 当前总条数: {count_result.count}")

client.close()

# 1291 + 457 = 1748