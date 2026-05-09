# Semantic Image Retrieval

Dự án này là một hệ thống Text Image Retrieval, sử dụng mô hình học sâu **CLIP** (của OpenAI) để trích xuất đặc trưng và **ChromaDB** để lưu trữ/tìm kiếm vector siêu tốc. 
Hệ thống cho phép tìm kiếm theo 2 chế độ với giao diện Web trực quan:
- **Text-to-Image**: Tìm ảnh bằng câu mô tả văn bản (VD: "a red apple", "a lion").
- **Image-to-Image**: Tìm ảnh tương đồng bằng cách upload ảnh truy vấn.

## Cấu trúc thư mục

```text
├── data/
│   ├── train/            # Dữ liệu ảnh train để build kho đặc trưng
│   └── test/             # Dữ liệu ảnh test dùng làm query
├── src/
│   ├── app.py            # Ứng dụng Web Giao diện (Streamlit)
│   ├── index_data.py     # Script lập chỉ mục toàn bộ ảnh vào Vector DB
│   ├── main.py           # Script chạy truy vấn ảnh cơ bản trên console (bản cũ)
│   ├── models/           # Các mô hình học sâu
│   │   ├── clip_model.py # Mô hình CLIP (OpenAI)
│   │   └── similarity.py # Các hàm tính độ tương đồng nguyên thủy
│   └── utils/            
│       ├── vector_db.py  # Cấu hình ChromaDB
│       ├── data_loader.py
│       └── visualization.py
├── requirements.txt      # Các thư viện phụ thuộc
└── .gitignore
```

## Hướng dẫn chạy

1. (Khuyến nghị) Khởi tạo và kích hoạt môi trường ảo bằng Conda:
   ```bash
   conda create -n datathon python=3.11 -y
   conda activate datathon
   ```

2. Cài đặt các thư viện yêu cầu:
   ```bash
   pip install -r requirements.txt
   ```

2. (Quan trọng) Lập chỉ mục Cơ sở dữ liệu:
   Bạn cần chạy lệnh này 1 lần duy nhất để hệ thống đọc ảnh trong thư mục `data/train` và trích xuất vector lưu vào ChromaDB:
   ```bash
   python src/index_data.py
   ```

3. Khởi chạy giao diện Web App:
   ```bash
   streamlit run src/app.py
   ```
   *Trình duyệt sẽ tự động mở ở địa chỉ `http://localhost:8501` để bạn thao tác trực tiếp.*
