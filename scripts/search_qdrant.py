from __future__ import annotations

import sys

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


COLLECTION_NAME = "traffic_law"
QDRANT_PATH = r"E:\RAG\db\db"
MODEL_NAME = "BAAI/bge-m3"


def main() -> None:
    client = QdrantClient(path=QDRANT_PATH)
    model = SentenceTransformer(MODEL_NAME)

    try:
        query = input("Nhập câu hỏi của bạn: ").strip()
        if not query:
            print("Câu hỏi không được để trống.")
            sys.exit(1)

        query_vector = model.encode(query, normalize_embeddings=True).tolist()
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=5,
            with_payload=True,
            with_vectors=False,
        )

        if not results.points:
            print("Không tìm thấy kết quả phù hợp.")
            return

        for index, point in enumerate(results.points, start=1):
            payload = point.payload or {}
            score = point.score

            print(f"\n#{index} score={score:.4f}" if score is not None else f"\n#{index} score=-")
            print(f"chunk_id: {payload.get('chunk_id', '-')}")
            print(f"chương: {payload.get('chương') or '-'}")
            print(f"điều: {payload.get('điều') or '-'}")
            print(f"tên_điều: {payload.get('tên_điều') or '-'}")
            print(f"khoản: {payload.get('khoản') or '-'}")
            print(f"ngữ_cảnh_đầy_đủ: {payload.get('ngữ_cảnh_đầy_đủ') or '-'}")
            print(payload.get("nội_dung", ""))
    finally:
        client.close()


if __name__ == "__main__":
    main()
