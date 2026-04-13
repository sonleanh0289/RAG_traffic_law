import json
import re
from pathlib import Path

from sentence_transformers import SentenceTransformer

def parse_chunks(file_path: Path) -> list[dict]:
    raw = file_path.read_text(encoding="utf-8").strip()
    blocks = re.split(r"\n\s*\n", raw)
    results = []

    for i, block in enumerate(blocks, start=1):
        context_match = re.search(r"Ngữ cảnh:\s*(.+)", block)
        content_match = re.search(r"Nội dung:\s*(.+)", block, re.DOTALL)

        if not context_match or not content_match:
            continue

        full_context = context_match.group(1).strip()
        content = content_match.group(1).strip()

        chương = None
        điều = None
        tên_điều = None
        khoản = None

        parts = [p.strip() for p in re.split(r"\s*-\s*", full_context)]
        if parts and parts[0].startswith("Chương"):
            chương = parts[0]

        for p in parts:
            if p.startswith("Điều"):
                khớp_điều = re.match(r"(Điều\s+\d+)\.\s*(.*)", p)
                if khớp_điều:
                    điều = khớp_điều.group(1).strip()
                    tên_điều = khớp_điều.group(2).strip() or None
                else:
                    điều = p.strip()
            elif p.startswith("Khoản"):
                khoản = p.strip()

        embed_text = f"{full_context}\n{content}"
        results.append(
            {
                "chunk_id": f"chunk_{i:03d}",
                "chương": chương,
                "điều": điều,
                "tên_điều": tên_điều,
                "khoản": khoản,
                "nội_dung": content,
                "ngữ_cảnh_đầy_đủ": full_context,
                "văn_bản_embed": embed_text,
            }
        )

    return results


def main() -> None:
    chunks = parse_chunks(file_path)
    if not chunks:
        raise ValueError(f"Không tìm thấy chunk hợp lệ trong {file_path}")

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
    print("Số chunk:", len(chunks))
    print("Số chiều vector:", len(chunks[0]["vector"]))
    print("Ví dụ chunk đầu:")
    print(chunks[0]["ngữ_cảnh_đầy_đủ"])
    print(chunks[0]["nội_dung"][:200])
    print("Vector sample:", chunks[0]["vector"][:8])

    file_output = Path("chunk_embeddings.json")
    file_output.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    
file_path = Path("E:\\RAG\\500 words chunk-20260325T170458Z-3-001\\500 words chunk\\chunk_001.txt")

if __name__ == "__main__":
    main()