import os
import numpy as np
import matplotlib.pyplot as plt

from utils import load_synthetic_data, plot_boundary
from decision_tree import gini, entropy
from svm_multiclass import MulticlassSVM
from adaboost_mh import AdaBoostMH, to_multilabel
from decision_tree import DecisionTreeClassifier

# Tạo thư mục lưu ảnh
os.makedirs('../figures', exist_ok=True)
os.makedirs('figures', exist_ok=True)

print("Đang chạy Thực nghiệm Cây quyết định (Decision Tree)...")
# Vẽ đồ thị so sánh impurity
p = np.linspace(0, 1, 300)
gini_vals = 2 * p * (1 - p)
entropy_vals = -(p * np.log2(p + 1e-12) + (1 - p) * np.log2(1 - p + 1e-12)) * 0.5
misclf_vals = 1 - np.maximum(p, 1 - p)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(p, gini_vals, label='Gini index', color='tab:red')
ax.plot(p, entropy_vals, label='Entropy (scaled x0.5)', color='tab:green', linestyle='--')
ax.plot(p, misclf_vals, label='Misclassification', color='black', linestyle=':')
ax.set_xlabel('Tỉ lệ lớp dương $p$')
ax.set_ylabel('Độ vẩn đục $F(n)$')
ax.set_title('So sánh ba tiêu chí độ vẩn đục nút (k=2)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/impurity_compare.pdf', dpi=150)
plt.savefig('../figures/impurity_compare.pdf', dpi=150) # Lưu thẳng vào thư mục báo cáo LaTeX
plt.close()

(X_tr_blob, X_te_blob, y_tr_blob, y_te_blob), _ = load_synthetic_data()

clf = DecisionTreeClassifier(criterion='gini', max_depth=4)
clf.fit(X_tr_blob, y_tr_blob)
dt_train_acc = np.mean(clf.predict(X_tr_blob) == y_tr_blob)
dt_test_acc  = np.mean(clf.predict(X_te_blob) == y_te_blob)
print(f" → Train accuracy: {dt_train_acc:.2%}")
print(f" → Test  accuracy: {dt_test_acc:.2%}")

print("Đang chạy Thực nghiệm Multi-class SVM...")
# Tải dữ liệu
(X_tr_blob, X_te_blob, y_tr_blob, y_te_blob), (X_tr_cls, X_te_cls, y_tr_cls, y_te_cls) = load_synthetic_data()

svm = MulticlassSVM(n_classes=3, lr=0.01, C=1.0, n_epochs=500)
svm.fit(X_tr_cls, y_tr_cls)

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(svm.loss_history, color='tab:blue', linewidth=1.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Hinge loss trung bình')
ax.set_title('Hội tụ của Multi-class SVM (subgradient descent)')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/svm_loss_curve.pdf', dpi=150)
plt.savefig('../figures/svm_loss_curve.pdf', dpi=150)
plt.close()

print("Đang chạy Thực nghiệm AdaBoost.MH...")
T_MAX = 50
F_vals, err_vals = [], []
Y_tr_ml = to_multilabel(y_tr_blob, k=3)

for t in range(1, T_MAX + 1):
    model_t = AdaBoostMH(T=t)
    model_t.fit(X_tr_blob, Y_tr_ml)
    
    # Tính F(alpha) gọn gàng bằng hàm predict_scores
    scores = model_t.predict_scores(X_tr_blob, k=3)
    F = np.sum(np.exp(-Y_tr_ml * scores))
    F_vals.append(F)
    
    pred = model_t.predict(X_tr_blob, k=3)
    err_vals.append(np.mean(pred != y_tr_blob))

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
plt.savefig('figures/adaboost_convergence.pdf', dpi=150)
plt.savefig('../figures/adaboost_convergence.pdf', dpi=150)
plt.close()

print("Hoàn tất! Các biểu đồ minh họa đã được tạo thành công.")