from qdrant_client import QdrantClient

# Giả sử các file của bạn nằm trong thư mục tên là 'my_legal_data'
# Bạn chỉ cần trỏ đường dẫn vào thư mục đó
client = QdrantClient(path="E:\RAG\db\db")

# Kiểm tra xem có đúng là dữ liệu đã ở đó không
collections = client.get_collections()
print("Các collection hiện có trong máy:", collections)

# Lấy thử một vài bản ghi để xem cấu trúc payload
# Thay 'ten_collection_cua_ban' bằng tên hiện ra ở lệnh trên
info = client.scroll(
    collection_name="nghi dinh 168",
    limit=5,
    with_payload=True,
    with_vectors=False
)
print("Dữ liệu mẫu:", info)

# 3. CHỦ ĐỘNG ĐÓNG KẾT NỐI (Tuyệt chiêu trị lỗi msvcrt trên Windows)
client.close()