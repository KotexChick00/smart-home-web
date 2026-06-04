# Face Recognition Module

Đây là module chịu trách nhiệm nhận diện khuôn mặt (Face Recognition) cho hệ thống Smart Home. Nó sử dụng mô hình học sâu để trích xuất đặc trưng (embedding) từ ảnh khuôn mặt và so khớp chúng bằng cơ sở dữ liệu vector tốc độ cao.

## Kiến trúc Hệ thống

- **Phát hiện khuôn mặt (Face Detection):** Sử dụng **RetinaFace** (thông qua package `uniface` từ module ngoài) để bắt vị trí khuôn mặt và trích xuất 5 điểm landmarks.
- **Trích xuất đặc trưng (Feature Extraction):** Sử dụng mạng **InceptionResnetV1** từ thư viện `facenet-pytorch`. Hệ thống sẽ convert ảnh về size `160x160` và trích xuất ra một vector `512` chiều.
- **Tìm kiếm vector (Vector Search):** Sử dụng **FAISS** (Facebook AI Similarity Search) với chuẩn đo khoảng cách `Cosine Similarity` (thông qua L2 Normalization + Inner Product) để tìm kiếm khuôn mặt khớp nhất trong chưa tới vài mili-giây.

## Cấu trúc Thư mục

```text
face_recognition/
├── model/                  # Chứa file Checkpoint và Database
│   ├── 20180408...pt       # Trọng số Casia-WebFace cho InceptionResnetV1
│   ├── faces.index         # Vector Database sinh bởi FAISS
│   └── users.pkl           # Danh sách Mapping giữa Vector ID và Tên user
├── src/                    # Chứa mã nguồn chính
│   ├── inference.py        # Logic trích xuất đặc trưng và quản lý FAISS DB
│   └── utils.py            # Chứa các hàm tiền xử lý ảnh (tiêu chuẩn Facenet)
├── tools/                  # Script hỗ trợ / Debug
│   ├── validate_lfw.py     # Script đánh giá model trên tập dataset LFW
│   ├── create_owner_db.py  # Script add mẫu thủ công vào DB
│   ├── add_samples.py      # Tương tự create_owner_db
│   └── inspect_weights.py  # Xem metadata trọng số model
└── README.md               # Tài liệu bạn đang đọc
```

## Quản lý Database & Reset Hệ thống

Toàn bộ thông tin đăng ký khuôn mặt của người dùng được lưu trong 2 file:
- `model/faces.index`: Bản đồ vector không gian (FAISS).
- `model/users.pkl`: Cấu trúc dữ liệu Python lưu thông tin Label/Tên định danh.

**Làm thế nào để Reset/Xóa toàn bộ người dùng?**
Bạn chỉ cần xóa (delete) 2 file `faces.index` và `users.pkl` trong thư mục `model/`. Ở lần chạy server tiếp theo, module sẽ tự động tạo lại một database trống tinh tươm.

## Dependencies

Module này yêu cầu các thư viện sau (đã nằm trong `requirements.txt` gốc):
- `torch`, `torchvision` (Chạy inference)
- `facenet-pytorch` (Cung cấp mạng InceptionResnetV1)
- `faiss-cpu` (hoặc `faiss-gpu` - Vector Database)
- `opencv-python` (Xử lý ảnh cơ bản)
- `numpy`
