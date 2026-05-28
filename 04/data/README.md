# Tài liệu Hướng dẫn Quản lý Dữ liệu (`data/`)

Thư mục `data/` được sử dụng để lưu trữ các bộ dữ liệu thực tế (Benchmark Datasets) phục vụ cho Giai đoạn đánh giá thực nghiệm của Đồ án Phân loại Đa lớp (Multi-class Classification).

---

## 1. Cơ chế Nạp dữ liệu Thông minh (Smart Loading & Simulation)

Để đảm bảo dự án luôn chạy mượt mà 100% trên mọi máy tính cá nhân mà không bắt buộc các thành viên phải tải về các file dữ liệu khổng lồ (hàng trăm MB đến hàng GB), hệ thống `src/utils/data_loader.py` được thiết kế với cơ chế kép:

1. **Ưu tiên nạp file vật lý (Physical Files):** Nếu hệ thống phát hiện các file dữ liệu vật lý (như `eurlex.libsvm` hoặc `mnist.csv`) nằm trong thư mục `data/`, nó sẽ tự động nạp trực tiếp qua các hàm chuẩn hóa của Scipy/Sklearn.
2. **Tự động mô phỏng (Simulation Fallback):** Nếu chưa có file vật lý, hệ thống sẽ tự động sử dụng bộ dữ liệu `load_digits` (phiên bản 8x8 thu gọn của MNIST) hoặc dùng `scipy.sparse` để mô phỏng chính xác cấu trúc ma trận thưa TF-IDF và phân phối mất cân bằng long-tail của EUR-Lex.

---

## 2. Chi tiết các Bộ dữ liệu Thực nghiệm

### 2.1. Bộ Dataset 1: Fashion-MNIST (Mono-label)

- **Vai trò:** Benchmark tiêu chuẩn vàng để kiểm chứng hiệu năng nền tảng và đối sánh chi phí giữa các chiến lược phân rã nhị phân.
- **Đặc tả dữ liệu:**
  - **Cấu trúc:** Ảnh Grayscale kích thước 28x28 (784 đặc trưng) hoặc 8x8 (64 đặc trưng).
  - **Số lớp (k):** 10 lớp
  - **Tính chất:** Dữ liệu mật độ dày (dense), cân bằng giữa các lớp.
- **Mục tiêu thực nghiệm:** So sánh trực tiếp chi phí thời gian huấn luyện và độ phức tạp giữa **One-Vs-All (10 models)** và **One-Vs-One (45 models)**.

### 2.2. Bộ Dataset 2: EUR-Lex / Eurlex-4K (Multi-label / Sparse)

- **Vai trò:** Thử thách nâng cao (Stress-test) kiểm chứng khả năng xử lý Extreme Multi-label Classification (XMC) và vấn đề nhãn đuôi (tail labels).
- **Đặc tả dữ liệu chuẩn (XMLRepository):**
  - **Train / Test:** 15,449 mẫu huấn luyện / 3,865 mẫu kiểm tra.
  - **Số đặc trưng:** 186,104 từ vựng (biểu diễn dưới dạng ma trận thưa TF-IDF).
  - **Số nhãn (Labels):** 3,956 chủ đề pháp lý Châu Âu (~5.3 nhãn / văn bản).
- **Giải thích định dạng LibSVM ("Tại sao dữ liệu toàn là số?"):**
  Dữ liệu EUR-Lex không lưu dưới dạng văn bản thô mà lưu dưới định dạng LibSVM để tiết kiệm hàng chục GB RAM. Một dòng tiêu biểu có cấu trúc:
  ```text
  12,45,102  5:0.123  18:0.456  1024:0.789
  ```

  - `12,45,102`: Danh sách ID các nhãn (chủ đề pháp lý) của văn bản.
  - `5:0.123`: Từ vựng thứ 5 trong từ điển xuất hiện với điểm số TF-IDF là `0.123`.

---

## 3. Hướng dẫn nạp file thủ công cho các thành viên

Nếu bạn muốn chạy thực nghiệm trên toàn bộ dữ liệu gốc quy mô lớn thay vì mô phỏng, vui lòng thực hiện theo các bước sau:

1. **Tải dữ liệu:** Truy cập vào kho lưu trữ của nhóm trên Google Drive theo đường dẫn sau: [Tải Data tại đây](https://drive.google.com/drive/folders/1uwnTxExtI4tOkHdnXVBSqheUEDAtGt97?usp=drive_link) và tải toàn bộ các file dữ liệu về.
2. **Lưu vào thư mục data:** Giải nén (nếu có) và đặt tất cả các file dữ liệu đã tải về vào thư mục `data/`.
3. **Cập nhật đường dẫn trong kịch bản:** Mở file `src/experiments/exp_real_datasets.py` và truyền đường dẫn file vào hàm tương ứng, ví dụ: `load_eurlex_data(file_path='data/eurlex_train.txt')`.

---

_Bản quyền © 2026 - Nhóm 04 - Đồ án Phân loại Đa lớp_
