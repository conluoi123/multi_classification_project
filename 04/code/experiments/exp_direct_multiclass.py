import sys
import os
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FIG_DIR = os.path.join(BASE_DIR, 'figures', 'direct_multiclass')
os.makedirs(FIG_DIR, exist_ok=True)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.utils.data_loader import load_synthetic_data
from src.utils.visualization import plot_boundary
from src.direct_multiclass.decision_tree import gini, entropy
from src.optimization.fast_tree import FastDecisionTreeClassifier
from src.direct_multiclass.svm_multiclass import MulticlassSVM
from src.optimization.memory_adaboost import MemoryEfficientAdaBoostMH as AdaBoostMH
from src.direct_multiclass.adaboost_mh import to_multilabel

def savefig(name):
    # Lưu định dạng gốc (.pdf dành cho báo cáo LaTeX/Overleaf)
    plt.savefig(os.path.join(FIG_DIR, name), dpi=150, bbox_inches='tight')
    
    # Tự động xuất thêm định dạng .png để tiện xem trực tiếp trên VS Code
    png_name = name.replace('.pdf', '.png')
    if png_name != name:
        plt.savefig(os.path.join(FIG_DIR, png_name), dpi=150, bbox_inches='tight')
        
    plt.close()

def run_decision_tree_experiment():
    print("Đang chạy Thực nghiệm Cây quyết định (Decision Tree)...")
    print("="*65)
    
    p = np.linspace(0, 1, 300)
    gini_vals    = 2 * p * (1 - p)
    entropy_vals = -(p * np.log2(p + 1e-12) + (1 - p) * np.log2(1 - p + 1e-12)) * 0.5
    misclf_vals  = 1 - np.maximum(p, 1 - p)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(p, gini_vals,    label='Gini index',            color='tab:red')
    ax.plot(p, entropy_vals, label='Entropy (scaled x0.5)', color='tab:green', linestyle='--')
    ax.plot(p, misclf_vals,  label='Misclassification',     color='black',     linestyle=':')
    ax.set_xlabel('Tỉ lệ lớp dương $p$')
    ax.set_ylabel('Độ vẩn đục $F(n)$')
    ax.set_title('So sánh ba tiêu chí độ vẩn đục nút (k=2)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig('impurity_compare.pdf')

    (X_tr_blob, X_te_blob, y_tr_blob, y_te_blob), _ = load_synthetic_data()
    clf = FastDecisionTreeClassifier(criterion='gini', max_depth=4)
    clf.fit(X_tr_blob, y_tr_blob)
    print(f"  → Train accuracy: {np.mean(clf.predict(X_tr_blob) == y_tr_blob):.2%}")
    print(f"  → Test  accuracy: {np.mean(clf.predict(X_te_blob) == y_te_blob):.2%}\n")

def run_multiclass_svm_experiment():
    print("Đang chạy Thực nghiệm Multi-class SVM...")
    print("="*65)

    (X_tr_blob, X_te_blob, y_tr_blob, y_te_blob), \
    (X_tr_cls,  X_te_cls,  y_tr_cls,  y_te_cls)  = load_synthetic_data()

    svm = MulticlassSVM(n_classes=3, lr=0.01, C=1.0, n_epochs=500)
    svm.fit(X_tr_cls, y_tr_cls)
    print(f"  → Train accuracy: {np.mean(svm.predict(X_tr_cls) == y_tr_cls):.2%}")
    print(f"  → Test  accuracy: {np.mean(svm.predict(X_te_cls) == y_te_cls):.2%}")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(svm.loss_history, color='tab:blue', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Hinge loss trung bình')
    ax.set_title('Hội tụ của Multi-class SVM (subgradient descent)')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig('svm_loss_curve.pdf')
    print("  → Đã lưu biểu đồ loss hội tụ.\n")

def run_adaboost_mh_experiment():
    print("Đang chạy Thực nghiệm AdaBoost.MH...")
    print("="*65)

    (X_tr_blob, X_te_blob, y_tr_blob, y_te_blob), _ = load_synthetic_data()
    T_MAX = 50
    F_vals, err_vals = [], []
    Y_tr_ml = to_multilabel(y_tr_blob, k=3)

    for t in range(1, T_MAX + 1):
        model_t = AdaBoostMH(T=t)
        model_t.fit(X_tr_blob, Y_tr_ml)
        scores = model_t.predict_scores(X_tr_blob, k=3)
        F_vals.append(np.sum(np.exp(-Y_tr_ml * scores)))
        err_vals.append(np.mean(model_t.predict(X_tr_blob, k=3) != y_tr_blob))

    ada_test_err = np.mean(model_t.predict(X_te_blob, k=3) != y_te_blob)
    print(f"  → Train accuracy: {1 - err_vals[-1]:.2%}")
    print(f"  → Test  accuracy: {1 - ada_test_err:.2%}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(range(1, T_MAX + 1), F_vals, color='tab:orange')
    ax1.set_xlabel('Số vòng lặp $T$')
    ax1.set_ylabel('$F(\\alpha)$')
    ax1.set_title('Hội tụ hàm mục tiêu AdaBoost.MH')
    ax1.grid(alpha=0.3)

    ax2.plot(range(1, T_MAX + 1), err_vals, color='tab:red')
    ax2.set_xlabel('Số vòng lặp $T$')
    ax2.set_ylabel('Training error')
    ax2.set_title('Training error theo số vòng lặp')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    savefig('adaboost_convergence.pdf')
    print("  → Đã lưu biểu đồ hội tụ AdaBoost.MH.\n")

if __name__ == "__main__":
    run_decision_tree_experiment()
    run_multiclass_svm_experiment()
    run_adaboost_mh_experiment()
    print("Hoàn tất! Các biểu đồ minh họa đã được tạo thành công.")
