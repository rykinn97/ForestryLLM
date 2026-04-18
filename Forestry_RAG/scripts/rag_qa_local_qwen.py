import os
import requests
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# =========================
# 0. 强制离线，禁止访问 Hugging Face
# =========================
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# =========================
# 1. 基本配置
# =========================

# 这里改成你服务器上真正的 Qdrant 本地库目录
qdrant_path = "/root/autodl-tmp/ForestryLLM/Forestry_RAG/qdrant_data"

collection_name = "forestry_kb"

# Ollama 本地 API
ollama_url = "http://127.0.0.1:11434/api/chat"
ollama_model = "qwen2.5:1.5b"

# 你本地缓存中的 bge-m3 路径
# 这里我用你截图里已经存在的 snapshot 路径
bge_model_path = "/root/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"

# =========================
# 2. 加载 embedding 模型（本地离线 + CPU）
# =========================
print("正在加载本地 embedding 模型...")
embed_model = SentenceTransformer(
    bge_model_path,
    device="cuda",
    local_files_only=True
)

# =========================
# 3. 连接 Qdrant
# =========================
print("正在连接 Qdrant...")
client = QdrantClient(path=qdrant_path)


# =========================
# 4. 检索函数
# =========================
def retrieve_context(question, top_k=5):
    query_vector = embed_model.encode(question, normalize_embeddings=True).tolist()

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True
    )

    retrieved_chunks = []
    for point in results.points:
        payload = point.payload
        retrieved_chunks.append({
            "score": point.score,
            "chunk_id": payload.get("chunk_id"),
            "book_title": payload.get("book_title"),
            "chapter_title": payload.get("chapter_title"),
            "section_title": payload.get("section_title"),
            "chunk_title": payload.get("chunk_title"),
            "cleaned_text": payload.get("cleaned_text"),
            "citation_anchor": payload.get("citation_anchor"),
        })
    return retrieved_chunks


# =========================
# 5. 构造提示词
# =========================
def build_messages(question, retrieved_chunks):
    context_text = ""
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_text += f"""[证据{i}]
标题：{chunk['chunk_title']}
来源：{chunk['citation_anchor']}
内容：{chunk['cleaned_text']}

"""

    system_prompt = """
你是一名林业专业知识问答助手。
请严格依据给定证据回答问题，不要脱离证据自由发挥。

要求：
1. 只能依据“给定证据”回答。
2. 如果证据不足，明确回答：“根据当前知识库证据，无法可靠回答该问题。”
3. 回答尽量简洁、准确、专业。
4. 回答后列出主要依据的引用来源。
"""

    user_prompt = f"""
用户问题：
{question}

给定证据：
{context_text}

请按下面格式输出：
答案：...
引用：...
"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


# =========================
# 6. 调用本地 Ollama Qwen
# =========================
def call_local_qwen(messages):
    payload = {
        "model": ollama_model,
        "messages": messages,
        "stream": False
    }

    response = requests.post(ollama_url, json=payload, timeout=300)
    response.raise_for_status()
    data = response.json()

    return data["message"]["content"]


# =========================
# 7. 主流程
# =========================
def ask_rag(question):
    chunks = retrieve_context(question, top_k=5)

    print("\n===== 检索到的证据 =====")
    for i, c in enumerate(chunks, 1):
        print(f"\n--- 证据{i} ---")
        print("score:", c["score"])
        print("标题:", c["chunk_title"])
        print("来源:", c["citation_anchor"])
        preview = c["cleaned_text"][:220]
        if len(c["cleaned_text"]) > 220:
            preview += "..."
        print("内容:", preview)

    messages = build_messages(question, chunks)
    answer = call_local_qwen(messages)

    print("\n===== 最终回答 =====")
    print(answer)


if __name__ == "__main__":
    try:
        question = input("请输入问题：").strip()
        ask_rag(question)
    finally:
        client.close()