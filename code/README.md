# Thực nghiệm Đồ án 2 - Nhập môn Học máy

Phần này chứa toàn bộ mã nguồn cài đặt từ đầu (from scratch) 3 thuật toán không kết hợp: Decision Tree, Multi-class SVM và AdaBoost.MH.

## Yêu cầu môi trường
* Python 3.9 trở lên.
* Cài đặt thư viện qua file `requirements.txt`.

## Cấu trúc thư mục
* `utils.py`: Hàm sinh dữ liệu tổng hợp và vẽ đồ thị.
* `decision_tree.py`: Cài đặt cây quyết định và các hàm impurity.
* `svm_multiclass.py`: Cài đặt SVM đa lớp bằng subgradient descent.
* `adaboost_mh.py`: Cài đặt AdaBoost.MH với Decision Stump.
* `run_experiments.py`: Kịch bản (script) chính để chạy huấn luyện và sinh biểu đồ minh họa.

## Hướng dẫn tái tạo kết quả
Mở terminal và thực thi các lệnh sau:
1. `pip install -r requirements.txt`
2. `python run_experiments.py`

Sau khi chạy xong, các biểu đồ minh họa (loss curve, impurity, hội tụ) sẽ được tự động xuất ra định dạng `.pdf` tại thư mục `figures/`.