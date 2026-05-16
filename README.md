# Đồ án 2: Phân loại Đa lớp (Multi-class Classification)

**Môn học:** Nhập môn Học máy (CSC14005)  
**Trường:** Đại học Khoa học Tự nhiên (ĐHQG-HCM)  
**Tài liệu tham khảo chính:** Sách *Foundations of Machine Learning (FML)* - Mohri et al.

---

## 1. Giới thiệu

Đồ án này tập trung vào việc nghiên cứu, cài đặt hoàn toàn từ đầu (from scratch) và kiểm chứng thực nghiệm các phương pháp phân loại đa lớp (Multi-class Classification) kinh điển. Hệ thống mã nguồn được tái cấu trúc theo mô hình **Clean Architecture** chuyên nghiệp, chia thành 2 nhóm chiến lược lớn:

### 🛡️ Nhóm 1: Các chiến lược Đa lớp dựa trên phân loại Nhị phân (`src/binary_reductions/`)
Sử dụng **Linear SVM** tự xây dựng làm bộ phân loại gốc (base learner) để triển khai:
* **One-Vs-All (OVA / OvR):** Huấn luyện $k$ bộ phân loại nhị phân (gồm bản gốc và bản hiệu chuẩn xác suất Calibrated OVA với Platt Scaling).
* **One-Vs-One (OVO):** Huấn luyện $k(k-1)/2$ bộ phân loại (gồm biểu quyết đa số kèm Tie-breaking và biểu quyết có trọng số Weighted OVO).
* **Error-Correcting Output Codes (ECOC):** Sử dụng ma trận mã hóa (Binary/Ternary) kết hợp giải mã Hard/Soft Decoding (Euclidean, Exponential Loss theo công thức 8.19 FML).

### ⚡ Nhóm 2: Các thuật toán Đa lớp trực tiếp (`src/direct_multiclass/`)
Cài đặt từ đầu 3 thuật toán giải quyết trực tiếp bài toán đa lớp không qua kết hợp:
* **Decision Tree (Cây quyết định):** Cài đặt cây quyết định đa lớp với tiêu chí Gini và Entropy, theo dõi độ vẩn đục qua các tầng sâu.
* **Multi-class SVM:** Tối ưu trực tiếp hàm mất mát Hinge loss đa lớp (primal form) bằng Subgradient Descent.
* **AdaBoost.MH:** Thuật toán Boosting đa lớp sử dụng ma trận nhãn đa nhãn $\{-1, +1\}^k$ kết hợp bộ phân loại yếu Decision Stump.

---

## 2. Thành viên thực hiện (Nhóm 04)

1. **Cao Quốc Tuấn** - 23120390   
2. **[Tên Thành viên 2]** - [MSSV 2]   
3. **[Tên Thành viên 3]** - [MSSV 3]   

*(Vui lòng cập nhật thông tin thành viên 2 và 3)*

---

## 3. Kiến trúc Hệ thống (Clean Architecture)

```text
multi_classification_project/
├── src/
│   ├── __init__.py
│   ├── base/                   # Các bộ Base Classifier dùng chung
│   │   ├── __init__.py
│   │   ├── binary_svm.py       # Bộ phân loại nhị phân gốc (Linear SVM)
│   │   └── decision_stump.py   # Base learner cho AdaBoost.MH
│   │
│   ├── binary_reductions/      # Nhóm 1: Các chiến lược phân rã nhị phân
│   │   ├── __init__.py
│   │   ├── ova_classifier.py   # Chiến lược OVA & Platt Scaling
│   │   ├── ovo_classifier.py   # Chiến lược OVO & Weighted OVO
│   │   └── ecoc_classifier.py  # Chiến lược ECOC (Hard/Soft decoding)
│   │
│   ├── direct_multiclass/      # Nhóm 2: Các thuật toán đa lớp trực tiếp
│   │   ├── __init__.py
│   │   ├── decision_tree.py    # Cây quyết định (Gini, Entropy)
│   │   ├── svm_multiclass.py   # Multi-class SVM (Subgradient descent)
│   │   └── adaboost_mh.py      # AdaBoost.MH
│   │
│   ├── utils/                  # Các module tiện ích dùng chung
│   │   ├── __init__.py
│   │   ├── data_loader.py      # Sinh và nạp dữ liệu tổng hợp
│   │   ├── visualization.py    # Vẽ ranh giới quyết định và cấu hình style
│   │   └── metrics.py          # Tính toán Accuracy, Confusion Matrix
│   │
│   └── experiments/            # Các kịch bản thực nghiệm được đóng gói
│       ├── __init__.py
│       ├── exp_binary_reductions.py
│       └── exp_direct_multiclass.py
│
├── figures/                    # Thư mục lưu trữ biểu đồ phân cấp tự động
│   ├── binary_reductions/      # Lưu 8 biểu đồ .png của Nhóm 1
│   └── direct_multiclass/      # Lưu 3 biểu đồ (song song .pdf & .png) của Nhóm 2
│
├── main.py                     # Master CLI Runner (Điều phối trung tâm)
├── requirements.txt            # Danh sách thư viện phụ thuộc chung
├── PROJECT_CONTEXT.md          # Tài liệu bối cảnh, tác dụng file và hạn chế
└── task.md                     # Danh sách công việc theo dõi tiến độ
```

---

## 4. Hướng dẫn sử dụng (Master CLI Runner)

### 4.1. Cài đặt môi trường
Đảm bảo bạn đã cài đặt Python 3.9 trở lên. Mở terminal tại thư mục gốc và thực thi lệnh sau để cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

### 4.2. Thực thi kịch bản thực nghiệm với `main.py`
Dự án sử dụng Master CLI `main.py` với `argparse` để điều phối toàn bộ các thực nghiệm một cách linh hoạt và chuyên nghiệp:

* **🚀 Chạy toàn bộ tất cả thực nghiệm của dự án (Khuyên dùng):**
  ```bash
  python main.py --all
  ```
  *(Hệ thống sẽ tự động tạo thư mục `figures/` và xuất mới 100% toàn bộ 11 biểu đồ minh họa)*

* **🛡️ Chạy riêng toàn bộ thực nghiệm Nhóm 1 (Binary Reductions):**
  ```bash
  python main.py --module binary_reductions
  ```

* **⚡ Chạy riêng toàn bộ thực nghiệm Nhóm 2 (Direct Multi-class):**
  ```bash
  python main.py --module direct_multiclass
  ```

* **🎯 Chạy một kịch bản thực nghiệm cụ thể (Ví dụ: Cây quyết định):**
  ```bash
  python main.py --experiment tree
  ```
  *(Các lựa chọn experiment hỗ trợ: `visual`, `ovo`, `performance`, `calibration`, `complexity`, `tree`, `svm`, `ad(aboost`)*

---

## 5. Danh mục Kết quả Thực nghiệm

Toàn bộ các biểu đồ minh họa và phân tích thực nghiệm được lưu trữ tập trung tại `figures/` theo đúng phân nhóm:

### 🛡️ Nhóm 1: Các chiến lược Đa lớp dựa trên Nhị phân (`figures/binary_reductions/`)
1. **Trực quan hóa Ma trận mã hóa (`all_coding_matrices.png`):** So sánh cấu trúc của OVA, OVO và ECOC.
2. **Decision Boundaries (`decision_boundaries_all.png`):** Ranh giới quyết định của các chiến lược trên dữ liệu 2D.
3. **Phân tích Lỗi OVO (`ovo_pairwise_heatmap.png`):** Ma trận nhầm lẫn cặp phân tích sai số giữa các lớp.
4. **Phân tích Hiệu năng (`performance_comparison.png`):** Bảng và biểu đồ so sánh Accuracy giữa 4 mô hình.
5. **ECOC Depth (`ecoc_depth_analysis.png`):** Chứng minh chiều dài mã $c$ tỷ lệ thuận với khả năng sửa lỗi và Accuracy.
6. **Platt Scaling (`calibration_impact.png`):** Hiệu quả nắn chỉnh xác suất trên dữ liệu mất cân bằng nặng.
7. **Kiểm chứng Lý thuyết (`complexity_time.png`):** Kiểm chứng độ phức tạp thời gian huấn luyện theo số lớp $k$ (Table 8.1 FML).
8. **Kiểm định Thống kê (`hypothesis_test.png`):** Paired T-test (10 runs) xác định sự khác biệt hiệu năng có ý nghĩa thống kê.

### ⚡ Nhóm 2: Các thuật toán Đa lớp trực tiếp (`figures/direct_multiclass/`)
*(💡 Hỗ trợ xuất song song định dạng `.pdf` chất lượng cao cho báo cáo LaTeX và `.png` để xem trực tiếp trên VS Code)*

1. **So sánh Tiêu chí Vẩn đục (`impurity_compare.pdf` / `.png`):** Đối sánh đồ thị giữa Gini index, Entropy và Misclassification error.
2. **Hội tụ Multi-class SVM (`svm_loss_curve.pdf` / `.png`):** Đường cong Hinge loss trung bình giảm dần qua các Epoch bằng Subgradient Descent.
3. **Hội tụ AdaBoost.MH (`adaboost_convergence.pdf` / `.png`):** Đồ thị hội tụ hàm mục tiêu $F(\alpha)$ và Training error giảm theo số vòng lặp $T$.

---

*Bản quyền © 2026 - Nhóm 04 - HCMUS*