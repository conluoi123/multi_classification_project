# Hướng dẫn chạy mã nguồn (Source Code Guide)

Thư mục `src/` chứa toàn bộ mã nguồn của dự án "Multi-class Classification" (Phân loại đa lớp). 

## 1. Môi trường cài đặt

Dự án sử dụng Python và các thư viện phổ biến trong lĩnh vực Machine Learning. 

**Cài đặt các thư viện cần thiết:**
Tại thư mục gốc của dự án, bạn có thể cài đặt các thư viện thông qua file `requirements.txt`:
```bash
pip install -r requirements.txt
```

Các thư viện chính được sử dụng bao gồm:
- `numpy >= 1.26.4`
- `scikit-learn >= 1.4.2`
- `matplotlib >= 3.8.4`
- `scipy >= 1.12.0`
- `pandas >= 2.2.0`
- `seaborn >= 0.13.2`

## 2. Hướng dẫn chạy code

Để chạy các thử nghiệm chính, bạn có thể thực thi trực tiếp file `main_experiments.py`. File này sẽ chạy tuần tự toàn bộ các kịch bản thử nghiệm bao gồm: trực quan hoá ma trận phân loại, kiểm chứng OVO chuyên sâu, đánh giá hiệu năng tổng thể, hiệu chuẩn (Platt Scaling) và độ phức tạp thời gian.

**Cú pháp:**
```bash
python src/main_experiments.py
```
*Lưu ý: Hãy chạy lệnh trên từ thư mục gốc của dự án (thư mục chứa thư mục `src`).*

Sau khi chạy xong, kết quả dạng Text sẽ hiển thị trên Terminal và các biểu đồ sẽ được tự động lưu vào thư mục `figures/` ở thư mục gốc.

Ngoài ra, bạn cũng có thể chạy các kịch bản thực nghiệm chuyên sâu được tổ chức riêng lẻ trong thư mục `experiments/`:
```bash
python src/experiments/exp_binary_reductions.py
python src/experiments/exp_real_datasets.py
# ...
```

## 3. Cấu trúc và mô tả ngắn từng thư mục/file

- **`main_experiments.py`**: Script tổng hợp thực thi các thử nghiệm thực hành, vẽ biểu đồ ranh giới quyết định, trực quan hóa phương pháp OVA/OVO/ECOC, kiểm thử sức mạnh mô hình, so sánh hiệu suất và thời gian.
- **`__init__.py`**: File đánh dấu thư mục `src` là một package Python.

### Các thư mục con:

- **`base/`**: Chứa các mô hình phân loại nhị phân cơ sở (Base Estimators).
  - `binary_svm.py`: Cài đặt thuật toán Support Vector Machine tuyến tính cho nhị phân (Binary Linear SVM).
  - `decision_stump.py`: Cài đặt Decision Stump, thường dùng làm bộ phân loại yếu (weak learner) cho thuật toán Boosting.

- **`binary_reductions/`**: Cài đặt các chiến lược quy hoạch bài toán đa lớp về thành nhiều bài toán nhị phân.
  - `ova_classifier.py`: Lược đồ One-vs-All (OVA) - Huấn luyện K bộ phân loại.
  - `ovo_classifier.py`: Lược đồ One-vs-One (OVO) - Huấn luyện K(K-1)/2 bộ phân loại.
  - `ecoc_classifier.py`: Lược đồ Error-Correcting Output Codes (ECOC) - Mã hoá lớp bằng ma trận sửa lỗi.

- **`direct_multiclass/`**: Cài đặt các mô hình giải quyết bài toán đa lớp một cách trực tiếp mà không qua quy hoạch nhị phân.
  - `adaboost_mh.py`: Thuật toán AdaBoost.MH cho phân loại đa lớp.
  - `decision_tree.py`: Cây quyết định (Decision Tree) hỗ trợ phân loại đa lớp trực tiếp.
  - `svm_multiclass.py`: SVM đa lớp (Crammer-Singer multiclass SVM).

- **`experiments/`**: Chứa các script thử nghiệm độc lập.
  - `exp_binary_reductions.py`: Chạy thực nghiệm tập trung vào các chiến lược quy hoạch (OVA, OVO, ECOC).
  - `exp_direct_multiclass.py`: Chạy thực nghiệm với các thuật toán phân loại đa lớp trực tiếp.
  - `exp_optimization_benchmark.py`: Benchmark so sánh tốc độ / hiệu suất của các phương pháp tối ưu hoá song song.
  - `exp_real_datasets.py`: Chạy thử nghiệm trên các tập dữ liệu thực tế (VD: MNIST, IRIS...).
  - `exp_tikz_compiler.py`: Hỗ trợ biên dịch sơ đồ cấu trúc và kết quả thực nghiệm ra định dạng TikZ (LaTex).

- **`optimization/`**: Cải tiến, tối ưu quá trình huấn luyện và dự đoán.
  - `fast_tree.py`: Tối ưu hoá cấu trúc cây quyết định.
  - `memory_adaboost.py`: Tối ưu hoá việc sử dụng bộ nhớ cho quá trình huấn luyện AdaBoost.
  - `parallel_ova.py`: Module huấn luyện song song (multiprocessing) tăng tốc độ cho OVA.
  - `parallel_ovo.py`: Module huấn luyện song song (multiprocessing) tăng tốc độ cho OVO.

- **`utils/`**: Các hàm tiện ích hỗ trợ tái sử dụng toàn cục.
  - `data_loader.py`: Hỗ trợ sinh dữ liệu giả (toy datasets) và tải dữ liệu chuẩn, thực hiện tiền xử lý.
  - `metrics.py`: Cung cấp các hàm đo lường hiệu suất (accuracy, tính toán matrix,...).
  - `visualization.py`: Tập hợp các hàm vẽ biểu đồ chuyên dụng (contour, heatmap, boundary...).
