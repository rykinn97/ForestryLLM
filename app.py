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
qdrant_path = "/root/autodl-tmp/ForestryLLM/Forestry_KB/qdrant_data"   # 改成你的实际路径
collection_name = "forestry_kb"

ollama_url = "http://127.0.0.1:11434/api/chat"
ollama_model = "qwen2.5:1.5b"

# 本地 bge-m3 路径
bge_model_path = "/root/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"

# 检索参数
top_k = 8
min_score = 0.68
max_context_chunks = 3
max_citations = 2   # 最多保留 2 条引用

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
# 4. 检查 Ollama 服务
# =========================
def check_ollama_service():
    try:
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=10)
        response.raise_for_status()
        return True
    except Exception as e:
        raise RuntimeError("无法连接到 Ollama 服务。请先在另一个终端运行：ollama serve") from e


# =========================
# 5. 检索函数（高分过滤）
# =========================
def retrieve_context(question, top_k=8, min_score=0.68, max_context_chunks=3):
    query_vector = embed_model.encode(question, normalize_embeddings=True).tolist()

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True
    )

    retrieved_chunks = []
    for point in results.points:
        if point.score < min_score:
            continue

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

    # 最多只保留前 max_context_chunks 条高分证据
    retrieved_chunks = retrieved_chunks[:max_context_chunks]

    return retrieved_chunks


# =========================
# 6. 构造提示词（模型只输出答案，不输出引用）
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
请严格依据给定证据回答问题。

必须遵守以下规则：
1. 只能依据“给定证据”中已经明确出现的信息回答。
2. 回答应当是对证据的直接概述或整理，不允许补充证据中没有的新信息。
3. 不要使用“可以推断”“结合可知”“说明了”“由此可见”等带推断味的表达。
4. 如果证据不足，必须原样输出：根据当前知识库证据，无法可靠回答该问题。
5. 回答尽量简洁、准确、专业。
6. 不要输出“引用：”“参考文献：”“证据：”等栏目，只输出答案正文。
"""

    user_prompt = f"""
用户问题：
{question}

给定证据：
{context_text}

请只输出答案正文，不要输出其他栏目。
"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


# =========================
# 7. 调用本地 Ollama Qwen
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

    return data["message"]["content"].strip()


# =========================
# 8. 自动生成引用（原封不动拷贝 citation_anchor）
# =========================
def build_citations(retrieved_chunks, max_citations=2):
    citations = []
    seen = set()

    for chunk in retrieved_chunks:
        anchor = chunk.get("citation_anchor")
        if not anchor:
            continue
        if anchor in seen:
            continue
        seen.add(anchor)
        citations.append(anchor)
        if len(citations) >= max_citations:
            break

    return citations


# =========================
# 9. 单次问答流程
# =========================
def ask_rag(question):
    chunks = retrieve_context(
        question,
        top_k=top_k,
        min_score=min_score,
        max_context_chunks=max_context_chunks
    )

    if not chunks:
        print("\n===== 最终回答 =====")
        print("答案：根据当前知识库证据，无法可靠回答该问题。")
        print("引用：无")
        return

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
    answer_text = call_local_qwen(messages)
    citations = build_citations(chunks, max_citations=max_citations)

    print("\n===== 最终回答 =====")
    print(f"答案：{answer_text}")
    print("引用：")
    if citations:
        for i, c in enumerate(citations, 1):
            print(f"{i}. {c}")
    else:
        print("无")


# =========================
# 10. 主程序：循环问答
# =========================
if __name__ == "__main__":
    try:
        check_ollama_service()

        print("系统已启动。输入问题开始问答。输入 quit / exit / q 可退出。")

        while True:
            question = input("\n请输入问题：").strip()

            if question.lower() in ["quit", "exit", "q"]:
                print("系统已退出。")
                break

            if not question:
                print("请输入有效问题。")
                continue

            ask_rag(question)

    finally:
        client.close()