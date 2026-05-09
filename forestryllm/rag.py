from __future__ import annotations

import requests
from typing import Any


INSUFFICIENT_EVIDENCE_ANSWER = "根据当前知识库证据，无法可靠回答该问题。"


def build_messages(question: str, retrieved_chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
    context_text = ""
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_text += (
            f"[证据{i}]\n"
            f"标题：{chunk.get('chunk_title')}\n"
            f"来源：{chunk.get('citation_anchor')}\n"
            f"内容：{chunk.get('cleaned_text')}\n\n"
        )

    system_prompt = (
        "你是一名林业专业知识问答助手。\n"
        "请严格依据给定证据回答问题。\n\n"
        "必须遵守以下规则：\n"
        "1. 只能依据“给定证据”中已经明确出现的信息回答。\n"
        "2. 回答应当是对证据的直接概述或整理，不允许补充证据中没有的新信息。\n"
        "3. 不要使用“可以推断”“结合可知”“说明了”“由此可见”等带推断味的表达。\n"
        f"4. 如果证据不足，必须原样输出：{INSUFFICIENT_EVIDENCE_ANSWER}\n"
        "5. 回答尽量简洁、准确、专业。\n"
        "6. 不要输出“引用：”“参考文献：”“证据：”等栏目，只输出答案正文。\n"
    )
    user_prompt = (
        f"用户问题：\n{question}\n\n"
        f"给定证据：\n{context_text}\n"
        "请只输出答案正文，不要输出其他栏目。"
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def call_ollama(config: dict[str, Any], messages: list[dict[str, str]]) -> str:
    generator = config["generator"]
    payload = {"model": generator["model"], "messages": messages, "stream": False}
    response = requests.post(generator["url"], json=payload, timeout=generator.get("timeout_seconds", 300))
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"].strip()


def build_citations(retrieved_chunks: list[dict[str, Any]], max_citations: int = 2) -> list[str]:
    citations: list[str] = []
    seen: set[str] = set()
    for chunk in retrieved_chunks:
        anchor = chunk.get("citation_anchor")
        if not anchor or anchor in seen:
            continue
        citations.append(anchor)
        seen.add(anchor)
        if len(citations) >= max_citations:
            break
    return citations
