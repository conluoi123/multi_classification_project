import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FIG_DIR = os.path.join(BASE_DIR, 'figures', 'real_datasets')
os.makedirs(FIG_DIR, exist_ok=True)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.utils.data_loader import load_mnist_data, load_eurlex_data
from src.base.binary_svm import BinaryLinearSVM
from src.optimization.parallel_ova import ParallelOneVsAllClassifier
from src.optimization.parallel_ovo import ParallelOneVsOneClassifier
from src.optimization.fast_tree import FastDecisionTreeClassifier

# Import các class gốc chưa tối ưu hóa để so sánh baseline
from src.binary_reductions.ova_classifier import OneVsAllClassifier
from src.binary_reductions.ovo_classifier import OneVsOneClassifier
from src.direct_multiclass.decision_tree import DecisionTreeClassifier

def savefig(name):
    plt.savefig(os.path.join(FIG_DIR, name), dpi=150, bbox_inches='tight')
    plt.close()

def run_mnist_experiment(use_optimization=True):
    print("="*70)
    mode_str = "TỐI ƯU HÓA (ĐA LUỒNG)" if use_optimization else "GỐC CHƯA TỐI ƯU (ĐƠN LUỒNG)"
    print(f" THỰC NGHIỆM TRÊN BỘ DỮ LIỆU THỰC TẾ 1: MNIST (MONO-LABEL) - {mode_str}")
    print("="*70)
    
    X_tr, X_te, y_tr, y_te = load_mnist_data(subsample=1000, test_size=0.2)
    
    OVAClass = ParallelOneVsAllClassifier if use_optimization else OneVsAllClassifier
    OVOClass = ParallelOneVsOneClassifier if use_optimization else OneVsOneClassifier
    
    print(f" Đang huấn luyện One-vs-All (OVA) với Linear SVM (10 models - {mode_str})...")
    start_ova = time.time()
    ova = OVAClass(base_estimator=BinaryLinearSVM(learning_rate=0.01, n_iters=100))
    ova.fit(X_tr, y_tr)
    time_ova = time.time() - start_ova
    acc_ova = np.mean(ova.predict(X_te) == y_te)
    print(f"  → OVA Time: {time_ova:.2f}s | Test Accuracy: {acc_ova:.2%}\n")

    print(f" Đang huấn luyện One-vs-One (OVO) với Linear SVM (45 models - {mode_str})...")
    start_ovo = time.time()
    ovo = OVOClass(base_estimator=BinaryLinearSVM(learning_rate=0.01, n_iters=100))
    ovo.fit(X_tr, y_tr)
    time_ovo = time.time() - start_ovo
    acc_ovo = np.mean(ovo.predict(X_te) == y_te)
    print(f"  → OVO Time: {time_ovo:.2f}s | Test Accuracy: {acc_ovo:.2%}\n")

    # Vẽ biểu đồ so sánh
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Biểu đồ thời gian
    bars1 = ax1.bar(['OVA (10 models)', 'OVO (45 models)'], [time_ova, time_ovo], color=['tab:blue', 'tab:orange'])
    ax1.set_ylabel('Thời gian huấn luyện (giây)')
    ax1.set_title(f'So sánh Chi phí Thời gian (MNIST - {mode_str})')
    ax1.bar_label(bars1, fmt='%.2fs')
    ax1.grid(alpha=0.3)

    # Biểu đồ Accuracy
    bars2 = ax2.bar(['OVA', 'OVO'], [acc_ova*100, acc_ovo*100], color=['tab:blue', 'tab:orange'])
    ax2.set_ylabel('Độ chính xác Test (%)')
    ax2.set_title(f'So sánh Hiệu năng (MNIST - {mode_str})')
    ax2.set_ylim(0, 100)
    ax2.bar_label(bars2, fmt='%.1f%%')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    prefix = 'opt_' if use_optimization else 'pure_'
    savefig(f'{prefix}mnist_ova_vs_ovo.png')
    print(f" Đã lưu biểu đồ so sánh MNIST tại: figures/real_datasets/{prefix}mnist_ova_vs_ovo.png\n")

def run_eurlex_experiment(use_optimization=True):
    print("="*70)
    mode_str = "TỐI ƯU HÓA (QUANTILE TREE)" if use_optimization else "GỐC CHƯA TỐI ƯU (PURE TREE)"
    print(f" THỰC NGHIỆM TRÊN BỘ DỮ LIỆU THỰC TẾ 2: EUR-LEX (MULTI-LABEL / SPARSE) - {mode_str}")
    print("="*70)
    
    # Nạp EUR-Lex từ file vật lý (nếu có trong data/Eurlex/) hoặc dùng mô phỏng ma trận thưa
    eurlex_file = os.path.join(BASE_DIR, 'data', 'Eurlex', 'eurlex_train.txt')
    X_tr, X_te, Y_tr, Y_te = load_eurlex_data(file_path=eurlex_file, n_samples=1000, n_features=200, n_classes=10)
    
    TreeClass = FastDecisionTreeClassifier if use_optimization else DecisionTreeClassifier
    
    # Chuyển đổi bài toán Multi-label thành việc huấn luyện Decision Tree trên từng lớp (Binary Relevance)
    print(f" Đang huấn luyện hệ thống phân loại đa nhãn (Extreme Classification) trên ma trận thưa ({mode_str})...")
    start_time = time.time()
    
    n_classes = Y_tr.shape[1]
    models = []
    for c in range(n_classes):
        clf = TreeClass(max_depth=3)
        clf.fit(X_tr, Y_tr[:, c])
        models.append(clf)
        
    train_time = time.time() - start_time
    
    # Đánh giá trên tập test (Hamming Loss và Exact Match)
    Y_pred = np.zeros_like(Y_te)
    for c in range(n_classes):
        Y_pred[:, c] = models[c].predict(X_te)
        
    hamming_loss = np.mean(Y_pred != Y_te)
    exact_match = np.mean(np.all(Y_pred == Y_te, axis=1))
    
    print(f"  → Thời gian huấn luyện ({n_classes} nhãn): {train_time:.2f}s")
    print(f"  → Hamming Loss (tỷ lệ lỗi nhãn): {hamming_loss:.2%}")
    print(f"  → Exact Match (đúng tuyệt đối 100% các nhãn): {exact_match:.2%}\n")

    # Vẽ biểu đồ phân phối nhãn dự đoán vs thực tế
    fig, ax = plt.subplots(figsize=(8, 4))
    true_counts = Y_te.sum(axis=0)
    pred_counts = Y_pred.sum(axis=0)
    
    x = np.arange(n_classes)
    width = 0.35
    ax.bar(x - width/2, true_counts, width, label='Thực tế (True Labels)', color='tab:green')
    ax.bar(x + width/2, pred_counts, width, label='Dự đoán (Predicted)', color='tab:purple')
    
    ax.set_xlabel('ID Chủ đề Pháp lý (Label ID)')
    ax.set_ylabel('Số lượng văn bản')
    ax.set_title(f'EUR-Lex: Phân phối Nhãn Thực tế vs Dự đoán ({mode_str})')
    ax.set_xticks(x)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    prefix = 'opt_' if use_optimization else 'pure_'
    savefig(f'{prefix}eurlex_performance.png')
    print(f"  Đã lưu biểu đồ EUR-Lex tại: figures/real_datasets/{prefix}eurlex_performance.png\n")

if __name__ == "__main__":
    run_mnist_experiment(use_optimization=True)
    run_eurlex_experiment(use_optimization=True)
