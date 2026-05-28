# Đồ án 2: Phân loại Đa lớp (Multi-class Classification)

## 1. Thông tin nhóm

Đồ án được thực hiện bởi **Nhóm 04**:

- Nguyễn Kim Quốc - 23120347
- Ngô Thị Thục Quyên - 23120348
- Cao Quốc Tuấn - 23120390
- Lục Hoàng Tuấn - 23120393
- Huỳnh Trọng Viên - 23120403

## 2. Chương sách đã chọn

Đồ án được thực hiện dựa trên **Chương 8: Multi-class Classification** trong sách giáo trình _Foundations of Machine Learning (FML)_ của nhóm tác giả Mehryar Mohri, Afshin Rostamizadeh, và Ameet Talwalkar.

## 3. Mô tả ngắn những gì đã làm và những điểm mở rộng so với sách

**Những gì đã làm:**

- **Cài đặt từ đầu (From scratch):** Tự xây dựng các bộ phân loại bằng Python/Numpy, sử dụng Linear SVM do nhóm tự cài đặt làm bộ phân loại cơ sở.
- **Chiến lược phân rã nhị phân:** Cài đặt các lược đồ One-Vs-All (OVA), One-Vs-One (OVO), và Error-Correcting Output Codes (ECOC).
- **Thuật toán đa lớp trực tiếp:** Triển khai Cây quyết định (Decision Tree), Multi-class SVM (tối ưu bằng Subgradient Descent) và AdaBoost.MH.
- **Kiến trúc mã nguồn:** Tổ chức source code chuyên nghiệp theo mô hình Clean Architecture, chia tách thành các module rõ ràng (`base`, `binary_reductions`, `direct_multiclass`, `optimization`, `utils`, `experiments`).

**Những điểm mở rộng so với sách:**

- **Tối ưu hóa đa luồng (Parallel Computing):** Triển khai huấn luyện song song (Parallel OVA, Parallel OVO) sử dụng thư viện đa luồng để tăng tốc độ huấn luyện với lượng dữ liệu lớn.
- **Tối ưu hóa bộ nhớ:** Cải tiến cài đặt AdaBoost.MH không sử dụng ma trận độn (tiết kiệm RAM) và áp dụng Quantile subsampling để tăng tốc thuật toán Cây quyết định.
- **Thử thách với Extreme Multi-label Classification (XMC):** Không chỉ dừng lại ở các dataset nhỏ lẻ, nhóm sử dụng bộ dữ liệu thực tế lớn EUR-Lex (hơn 186.000 đặc trưng, gần 4000 nhãn pháp lý phân phối mất cân bằng) kết hợp cấu trúc ma trận thưa (Sparse Matrix).
- **Trực quan hóa phân tích sâu:** Xây dựng các biểu đồ phân tích lỗi chuyên sâu (Pairwise Confusion Heatmap cho OVO, đồ thị so sánh độ sâu mã ECOC) và công cụ tự động biên dịch kết quả ra định dạng TikZ cho báo cáo LaTeX.
- **Nâng cấp thuật toán:** Tích hợp thêm kỹ thuật hiệu chuẩn xác suất Platt Scaling cho OVA và cơ chế biểu quyết có trọng số (Weighted Voting) cho OVO.

## 4. Hướng dẫn tái tạo kết quả thực nghiệm

Để chạy lại toàn bộ các thử nghiệm và tái tạo kết quả, bạn vui lòng thực hiện theo các bước sau:

> **LƯU Ý VỀ DỮ LIỆU THỰC TẾ:**
> Để đảm bảo quá trình chạy thử (demo) diễn ra nhanh chóng, mượt mà và không gây tràn RAM (Out-of-Memory) trên máy tính cá nhân, hệ thống mặc định được thiết lập để chỉ trích xuất và chạy trên một **tập con nhỏ (subsample)** của các bộ dữ liệu khổng lồ (ví dụ: MNIST giới hạn ở 1000 ảnh, EUR-Lex giới hạn ở 1000 văn bản).
>
> _Nếu máy tính có cấu hình mạnh và bạn muốn huấn luyện trên 100% dữ liệu gốc, bạn có thể chỉnh sửa các tham số `subsample=None` hoặc `n_samples` trong file `src/experiments/exp_real_datasets.py`._

**Bước 1: Cài đặt môi trường**
Đảm bảo máy tính đã cài đặt Python (phiên bản 3.9 trở lên). Mở terminal tại thư mục gốc của dự án và chạy lệnh:

```bash
pip install -r requirements.txt
```

**Bước 2: Thực thi kịch bản (Master CLI Runner)**
Dự án sử dụng file `main.py` để điều phối mọi thử nghiệm một cách tự động. Các biểu đồ kết quả sẽ được sinh ra và lưu tại thư mục `figures/`.

- **Chạy toàn bộ thực nghiệm từ A-Z (Khuyên dùng):**

  ```bash
  python main.py --all
  ```

  Hệ thống sẽ chạy liên tiếp toàn bộ các kịch bản thuật toán từ dữ liệu mô phỏng, trực quan ranh giới quyết định, cho đến dữ liệu thực tế (MNIST, EUR-Lex) và lưu lại kết quả.

- **Chạy theo từng nhóm chiến lược (Modules):**
  - Chỉ chạy nhóm phân rã nhị phân: `python main.py --module binary_reductions`
  - Chỉ chạy nhóm đa lớp trực tiếp: `python main.py --module direct_multiclass`
  - Chạy nhóm dữ liệu thực tế: `python main.py --module real_datasets`
  - Chạy các module tối ưu đa luồng: `python main.py --optimize`
  - Chạy module biên dịch biểu đồ LaTeX (TikZ): `python main.py --tikz`

- **Chạy từng thuật toán hoặc thử nghiệm cụ thể lẻ (Experiments):**
  Ví dụ, nếu chỉ muốn kiểm chứng phân tích lỗi của mô hình OVO hoặc chạy thử nghiệm Cây quyết định:
  ```bash
  python main.py --experiment ovo
  python main.py --experiment tree
  python main.py --experiment eurlex
  ```
