from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from .config import resolve_path


def build_index(config: dict[str, Any], records: list[dict[str, Any]], recreate: bool = False) -> dict[str, Any]:
    model_path = resolve_path(config["embedding_model_path"])
    qdrant_path = resolve_path(config["qdrant_path"])
    collection_name = config["collection_name"]
    device = config.get("embedding_device", "cpu")

    model = SentenceTransformer(str(model_path), device=device, local_files_only=True)
    texts = [str(record["cleaned_text"]) for record in records]
    embeddings = model.encode(texts, batch_size=16, show_progress_bar=True, normalize_embeddings=True)
    vector_size = len(embeddings[0])

    client = QdrantClient(path=str(qdrant_path))
    try:
        collection_names = [c.name for c in client.get_collections().collections]
        if collection_name in collection_names and recreate:
            client.delete_collection(collection_name=collection_name)
            collection_names.remove(collection_name)
        if collection_name not in collection_names:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

        points = []
        for idx, (record, vector) in enumerate(zip(records, embeddings)):
            payload = {key: record.get(key) for key in [
                "chunk_id",
                "book_title",
                "chapter_title",
                "section_title",
                "page_start",
                "page_end",
                "chunk_title",
                "topic_type",
                "cleaned_text",
                "keywords",
                "citation_anchor",
            ]}
            points.append(PointStruct(id=idx, vector=vector.tolist(), payload=payload))

        client.upsert(collection_name=collection_name, points=points)
        return {
            "collection_name": collection_name,
            "qdrant_path": str(qdrant_path),
            "records_indexed": len(records),
            "vector_size": vector_size,
            "embedding_model_path": str(model_path),
        }
    finally:
        client.close()


def qdrant_search(config: dict[str, Any], question: str, top_k: int) -> list[dict[str, Any]]:
    model_path = resolve_path(config["embedding_model_path"])
    qdrant_path = resolve_path(config["qdrant_path"])
    collection_name = config["collection_name"]
    device = config.get("embedding_device", "cpu")

    model = SentenceTransformer(str(model_path), device=device, local_files_only=True)
    query_vector = model.encode(question, normalize_embeddings=True).tolist()
    client = QdrantClient(path=str(qdrant_path))
    try:
        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        output = []
        for point in results.points:
            payload = point.payload or {}
            output.append({
                "score": point.score,
                "chunk_id": payload.get("chunk_id"),
                "book_title": payload.get("book_title"),
                "chapter_title": payload.get("chapter_title"),
                "section_title": payload.get("section_title"),
                "chunk_title": payload.get("chunk_title"),
                "cleaned_text": payload.get("cleaned_text"),
                "citation_anchor": payload.get("citation_anchor"),
                "topic_type": payload.get("topic_type"),
            })
        return output
    finally:
        client.close()
