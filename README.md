# Đồ án 2: Phân loại Đa lớp (Multi-class Classification)
**Môn học:** Nhập môn Học máy (CSC14005)  
**Trường:** Đại học Khoa học Tự nhiên (ĐHQG-HCM)


## 1. Giới thiệu
Đồ án này tập trung vào việc cài đặt từ đầu (from scratch) và kiểm chứng thực nghiệm các chiến lược phân loại đa lớp phổ biến dựa trên bộ phân loại nhị phân (Binary Classifiers). Các chiến lược bao gồm:
* **One-Vs-All (OVA):** Huấn luyện $k$ bộ phân loại, mỗi bộ cho một lớp.
* **One-Vs-One (OVO):** Huấn luyện $k(k-1)/2$ bộ phân loại cho mọi cặp lớp có thể.
* **Error-Correcting Output Codes (ECOC):** Sử dụng ma trận mã hóa để tăng khả năng sửa lỗi và độ chính xác.

## 2. Thành viên thực hiện -
1. **Cao Quốc Tuấn** - 23120390   
2. **Cao Quốc Tuấn** - 23120390   
3. **Cao Quốc Tuấn** - 23120390   
---

## 3. Cấu trúc thư mục
```text
multi_classification_project/
├── src/                        # Chứa mã nguồn thuật toán
│   ├── __init__.py             # Khai báo package
│   ├── base_svm.py             # Bộ phân loại nhị phân gốc (SVM)
│   ├── ova_classifier.py       # Chiến lược One-Vs-All
│   ├── ovo_classifier.py       # Chiến lược One-Vs-One
│   ├──  main_experiments.py    # File chạy thực nghiệm chính
│   └── ecoc_classifier.py      # Chiến lược ECOC
├── figures/                    # Chứa các biểu đồ kết quả (tự động sinh ra)
├── .gitignore                  # Cấu hình bỏ qua các file rác
├── requirements.txt            # Danh sách thư viện cần thiết
└── README.md                   

```

---

## 4. Hướng dẫn sử dụng

### Cài đặt môi trường

Đảm bảo bạn đã cài đặt Python 3.8 trở lên. Sử dụng lệnh sau để cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt

```

### Chạy thực nghiệm

Để tái lập toàn bộ kết quả và sinh ra các biểu đồ trong báo cáo, hãy chạy lệnh:

```bash
python src/main_experiments.py

```

---

## 5. Các kết quả đạt được

Hệ thống sẽ tự động thực hiện và xuất ra các kết quả tại thư mục `figures/`:

1. **Trực quan hóa Ma trận mã hóa:** So sánh cấu trúc của OVA, OVO và ECOC.
2. **Decision Boundaries:** Ranh giới quyết định của các chiến lược trên dữ liệu 2D.
3. **Phân tích Hiệu năng:** So sánh Accuracy giữa các mô hình.
4. **Kiểm chứng Lý thuyết:** * Kiểm chứng độ phức tạp thời gian huấn luyện theo số lớp $k$ (Table 8.1 FML).
* Phân tích khả năng sửa lỗi của ECOC dựa trên chiều dài mã.
* Đánh giá hiệu quả của Platt Scaling (Calibration) trên dữ liệu mất cân bằng.


5. **Kiểm định thống kê:** Thực hiện Paired T-test để xác định sự khác biệt có ý nghĩa.

---

*Bản quyền © 2026 - Nhóm 04 - HCMUS*