from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from embedding_models import parse_chunks
from sentence_transformers import SentenceTransformer


#client = QdrantClient("http://localhost:6333")
client = QdrantClient(path=r"E:\RAG\qdrant_data")
collection_name = "traffic_law"
folder_path = Path("E:\\RAG\\500 words chunk-20260325T170458Z-3-001\\500 words chunk")
chunks = []

for path in folder_path.glob("*.txt"):
    chunks.extend(parse_chunks(path))

if not chunks:
    raise ValueError(f"Không tìm thấy chunk hợp lệ trong {folder_path}")

for i, chunk in enumerate(chunks, start=1):
    chunk["chunk_id"] = f"chunk_{i:03d}"

model = SentenceTransformer("BAAI/bge-m3")
texts = [chunk["văn_bản_embed"] for chunk in chunks]

các_vector = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True,
    batch_size=16,
)

for chunk, vector in zip(chunks, các_vector, strict=True):
    chunk["vector"] = vector.tolist()

vector_size = len(chunks[0]["vector"])

if client.collection_exists(collection_name):
    collection_info = client.get_collection(collection_name)
    current_vectors = collection_info.config.params.vectors

    if current_vectors.size != vector_size:
        raise ValueError(
            f"Collection '{collection_name}' đang có vector size {current_vectors.size}, "
            f"không khớp với vector mới {vector_size}"
        )

    if current_vectors.distance != Distance.COSINE:
        raise ValueError(
            f"Collection '{collection_name}' đang dùng distance {current_vectors.distance}, "
            "không khớp với COSINE"
        )
else:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        )
    )

points = []
for i, chunk in enumerate(chunks):
    points.append(
        PointStruct(
            id=i,
            vector=chunk["vector"],
            payload={
                "chunk_id": chunk["chunk_id"],
                "chương": chunk["chương"],
                "điều": chunk["điều"],
                "tên_điều": chunk["tên_điều"],
                "khoản": chunk["khoản"],
                "nội_dung": chunk["nội_dung"],
                "ngữ_cảnh_đầy_đủ": chunk["ngữ_cảnh_đầy_đủ"],
                "văn_bản_embed": chunk["văn_bản_embed"],
            },
        )
    )

result = client.upsert(
    collection_name=collection_name,
    points=points,
    wait=True,
)
print(result)
print(f"Đã upsert {len(points)} points vào collection '{collection_name}'")
print("After:", client.get_collections())
print("traffic_law exists after:", client.collection_exists(collection_name))
client.close()
