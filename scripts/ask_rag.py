from __future__ import annotations

import sys

from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


COLLECTION_NAME = "nghi dinh 168"
QDRANT_PATH = r"E:\RAG\db\db"
MODEL_NAME = "BAAI/bge-m3"

LLM_BASE_URL = "http://localhost:11434/v1"
LLM_API_KEY = "ollama"
LLM_MODEL = "qwen2.5:7b"
TOP_K = 5
SYSTEM_PROMPT = """Bạn là một trợ lý pháp lý am hiểu luật giao thông Việt Nam. Bạn sẽ được cung cấp một câu hỏi và một số đoạn văn bản liên quan từ luật giao thông đường bộ Việt Nam. Nhiệm vụ của bạn là đọc kỹ các đoạn văn bản này và trả lời câu hỏi dựa trên thông tin đã cho. Hãy đảm bảo rằng câu trả lời của bạn chính xác, rõ ràng và dựa trên nội dung của các đoạn văn bản được cung cấp."""

#db cua son
# def build_context(points) -> str:
#     blocks = []
#     for i, point in enumerate(points, start=1):
#         payload = point.payload or {}
#         blocks.append(
#             f"[Tài liệu {i}]\n"
#             f"Ngữ cảnh: {payload.get('ngữ_cảnh_đầy_đủ', '-')}\n"
#             f"Nội dung: {payload.get('nội_dung', '-')}"
#         )
#     return "\n\n".join(blocks)

#db cua thanh
def build_context(points) -> str:
    blocks = []
    for i, point in enumerate(points, start=1):
        payload = point.payload or {}
        metadata = payload.get("metadata") or {}

        blocks.append(
            f"[Tài liệu {i}]\n"
            f"Nguồn: {metadata.get('source', '-')}\n"
            f"{payload.get('page_content', '-')}"
        )

    return "\n\n".join(blocks)


def build_prompt(question: str, points) -> str:
    context = build_context(points)
    return f"""CONTEXT:
{context}

CÂU HỎI:
{question}

HÃY TRẢ LỜI CÂU HỎI DỰA TRÊN NGỮ CẢNH VÀ NỘI DUNG ĐƯỢC CUNG CẤP Ở TRÊN. NẾU KHÔNG TÌM THẤY THÔNG TIN LIÊN QUAN TRONG NGỮ CẢNH, HÃY TRẢ LỜI RẰNG "KHÔNG TÌM THẤY THÔNG TIN LIÊN QUAN"."""


def retrieve(question: str, client: QdrantClient, model: SentenceTransformer, limit: int = 5):
    query_vector = model.encode(question, normalize_embeddings=True).tolist()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return results.points


def ask_llm(question: str, points) -> str:
    prompt = build_prompt(question, points)
    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content
    return content.strip() if isinstance(content, str) else str(content)


def main() -> None:
    qdrant_client = QdrantClient(path=QDRANT_PATH)
    model = SentenceTransformer(MODEL_NAME)

    try:
        if not qdrant_client.collection_exists(COLLECTION_NAME):
            print(f"Không tìm thấy collection '{COLLECTION_NAME}'")
            sys.exit(1)

        question = input("Nhập câu hỏi của bạn: ").strip()
        if not question:
            print("Câu hỏi không được để trống.")
            sys.exit(1)

        points = retrieve(question, qdrant_client, model, limit=TOP_K)
        if not points:
            print("Không truy xuất được context nào.")
            sys.exit(1)

        print("\n=== TOP-K RETRIEVAL ===")
        for i, point in enumerate(points, start=1):
            payload = point.payload or {}
            print(f"\n[{i}] score={point.score:.4f}" if point.score is not None else f"\n[{i}] score=-")
            print(payload.get("page_content", "-"))
            print(payload.get("metadata", {}).get("source", "-")[:300])

        print("\n=== LLM ANSWER ===")
        try:
            answer = ask_llm(question, points)
        except Exception as exc:
            print(f"Không gọi được local LLM: {exc}")
            sys.exit(1)

        print(answer)
    finally:
        qdrant_client.close()


if __name__ == "__main__":
    main()
