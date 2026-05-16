# Đồ án 2: Phân loại Đa lớp (Multi-class Classification)

**Môn học:** Nhập môn Học máy (CSC14005)  
**Trường:** Đại học Khoa học Tự nhiên (ĐHQG-HCM)  
**Tài liệu tham khảo chính:** Sách *Foundations of Machine Learning (FML)* - Mohri et al.

---

## 1. Giới thiệu

Đồ án này tập trung vào việc nghiên cứu, cài đặt từ đầu (from scratch) và kiểm chứng thực nghiệm các phương pháp phân loại đa lớp (Multi-class Classification) kinh điển. Hệ thống được chia thành 2 nhóm chiến lược lớn:

### Nhóm 1: Các chiến lược Đa lớp dựa trên phân loại Nhị phân (`src/`)
Sử dụng **Linear SVM** tự xây dựng làm bộ phân loại gốc để triển khai:
* **One-Vs-All (OVA / OvR):** Huấn luyện $k$ bộ phân loại nhị phân (gồm bản gốc và bản hiệu chuẩn xác suất Calibrated OVA với Platt Scaling).
* **One-Vs-One (OVO):** Huấn luyện $k(k-1)/2$ bộ phân loại (gồm biểu quyết đa số kèm Tie-breaking và biểu quyết có trọng số Weighted OVO).
* **Error-Correcting Output Codes (ECOC):** Sử dụng ma trận mã hóa (Binary/Ternary) kết hợp giải mã Hard/Soft Decoding (Euclidean, Exponential Loss theo công thức 8.19 FML).

### Nhóm 2: Các thuật toán Đa lớp trực tiếp (`code/`)
Cài đặt từ đầu 3 thuật toán giải quyết đa lớp không qua kết hợp:
* **Decision Tree (Cây quyết định):** Cài đặt cây quyết định đa lớp với tiêu chí Gini và Entropy, theo dõi vẩn đục qua các tầng sâu.
* **Multi-class SVM:** Tối ưu trực tiếp hàm mất mát Hinge loss đa lớp (primal form) bằng Subgradient Descent.
* **AdaBoost.MH:** Thuật toán Boosting đa lớp sử dụng ma trận nhãn đa nhãn $\{-1, +1\}^k$ kết hợp bộ phân loại yếu Decision Stump.

---

## 2. Thành viên thực hiện (Nhóm 04)

1. **Cao Quốc Tuấn** - 23120390   
2. **[Tên Thành viên 2]** - [MSSV 2]   
3. **[Tên Thành viên 3]** - [MSSV 3]   

*(Vui lòng cập nhật thông tin thành viên 2 và 3)*

---

## 3. Cấu trúc thư mục hiện tại

```text
multi_classification_project/
├── src/                        # Phần 1: Chiến lược Đa lớp dựa trên Nhị phân
│   ├── __init__.py
│   ├── base_svm.py             # Bộ phân loại nhị phân gốc (Linear SVM)
│   ├── ova_classifier.py       # Chiến lược OVA & Platt Scaling
│   ├── ovo_classifier.py       # Chiến lược OVO & Weighted OVO
│   ├── ecoc_classifier.py      # Chiến lược ECOC (Binary/Ternary, Hard/Soft decoding)
│   └── main_experiments.py     # Kịch bản thực nghiệm Phần 1
├── code/                       # Phần 2: Thuật toán Đa lớp trực tiếp
│   ├── utils.py                # Sinh dữ liệu tổng hợp và vẽ Decision Boundary
│   ├── decision_tree.py        # Cây quyết định (Gini, Entropy, Impurity tracking)
│   ├── svm_multiclass.py       # Multi-class SVM (Subgradient descent)
│   ├── adaboost_mh.py          # AdaBoost.MH & Decision Stump
│   └── run_experiments.py      # Kịch bản thực nghiệm Phần 2
├── figures/                    # Thư mục lưu trữ biểu đồ xuất ra tự động (.png & .pdf)
├── requirements.txt            # Danh sách thư viện phụ thuộc chung
├── README.md                   # Tài liệu hướng dẫn sử dụng chính
├── PROJECT_CONTEXT.md          # Tài liệu tổng hợp bối cảnh, tác dụng file và hạn chế
└── task.md                     # Danh sách công việc theo dõi tiến độ tái cấu trúc
```

---

## 4. Hướng dẫn sử dụng

### 4.1. Cài đặt môi trường
Đảm bảo bạn đã cài đặt Python 3.9 trở lên. Mở terminal tại thư mục gốc và thực thi lệnh sau để cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

### 4.2. Chạy thực nghiệm
Dự án cung cấp 2 kịch bản thực nghiệm độc lập tương ứng với 2 phần:

* **Chạy thực nghiệm Phần 1 (Các chiến lược dựa trên Nhị phân):**
  ```bash
  python src/main_experiments.py
  ```
  *(Hệ thống sẽ tự động xuất 8 biểu đồ định dạng `.png` vào thư mục `figures/`)*

* **Chạy thực nghiệm Phần 2 (Các thuật toán Đa lớp trực tiếp):**
  ```bash
  python code/run_experiments.py
  ```
  *(Hệ thống sẽ tự động xuất 3 biểu đồ định dạng `.pdf` vào thư mục `figures/`)*

---

## 5. Các kết quả đạt được

Toàn bộ các biểu đồ minh họa và phân tích thực nghiệm được lưu trữ tập trung tại `figures/`:

### Phần 1: Chiến lược Đa lớp dựa trên Nhị phân
1. **Trực quan hóa Ma trận mã hóa (`all_coding_matrices.png`):** So sánh cấu trúc của OVA, OVO và ECOC.
2. **Decision Boundaries (`decision_boundaries_all.png`):** Ranh giới quyết định của các chiến lược trên dữ liệu 2D.
3. **Phân tích Lỗi OVO (`ovo_pairwise_heatmap.png`):** Ma trận nhầm lẫn cặp phân tích sai số giữa các lớp.
4. **Phân tích Hiệu năng (`performance_comparison.png`):** Bảng và biểu đồ so sánh Accuracy giữa 4 mô hình.
5. **ECOC Depth (`ecoc_depth_analysis.png`):** Chứng minh chiều dài mã $c$ tỷ lệ thuận với khả năng sửa lỗi và Accuracy.
6. **Platt Scaling (`calibration_impact.png`):** Hiệu quả nắn chỉnh xác suất trên dữ liệu mất cân bằng nặng.
7. **Kiểm chứng Lý thuyết (`complexity_time.png`):** Kiểm chứng độ phức tạp thời gian huấn luyện theo số lớp $k$ (Table 8.1 FML).
8. **Kiểm định Thống kê (`hypothesis_test.png`):** Paired T-test (10 runs) xác định sự khác biệt hiệu năng có ý nghĩa thống kê.

### Phần 2: Thuật toán Đa lớp trực tiếp
1. **So sánh Tiêu chí Vẩn đục (`impurity_compare.pdf`):** Đối sánh đồ thị giữa Gini index, Entropy và Misclassification error.
2. **Hội tụ Multi-class SVM (`svm_loss_curve.pdf`):** Đường cong Hinge loss trung bình giảm dần qua các Epoch bằng Subgradient Descent.
3. **Hội tụ AdaBoost.MH (`adaboost_convergence.pdf`):** Đồ thị hội tụ hàm mục tiêu $F(\alpha)$ và Training error giảm theo số vòng lặp $T$.

---

*Bản quyền © 2026 - Nhóm 04 - HCMUS*