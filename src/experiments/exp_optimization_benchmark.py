import sys
import os
import time
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_blobs
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_rel

# CẤU HÌNH HỆ THỐNG VÀ ĐƯỜNG DẪN
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('ggplot')
plt.rcParams.update({'font.size': 10, 'figure.dpi': 120})
np.random.seed(42)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FIG_DIR = os.path.join(BASE_DIR, 'figures', 'optimization_benchmark')
os.makedirs(FIG_DIR, exist_ok=True)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.base.binary_svm import BinaryLinearSVM
from src.optimization.parallel_ova import ParallelOneVsAllClassifier
from src.optimization.parallel_ovo import ParallelOneVsOneClassifier
from src.optimization.memory_adaboost import MemoryEfficientAdaBoostMH
from src.optimization.fast_tree import FastDecisionTreeClassifier
from src.binary_reductions.ecoc_classifier import ECOCClassifier
from src.binary_reductions.ova_classifier import CalibratedOneVsAllClassifier
from src.direct_multiclass.adaboost_mh import to_multilabel

def run_optimized_rich_analysis():
    print("="*70)
    print(" BẮT ĐẦU CHẠY BỘ PHÂN TÍCH ĐỈNH CAO (RICH ANALYSIS) - PHIÊN BẢN TỐI ƯU")
    print("="*70)
    start_total = time.time()

    # ---------------------------------------------------------
    # PHẦN 1: TRỰC QUAN HÓA RANH GIỚI QUYẾT ĐỊNH (ĐA LUỒNG)
    # ---------------------------------------------------------
    print("\n[1/5] Đang huấn luyện mô hình Đa luồng để vẽ Ranh giới quyết định (Decision Boundaries)...")
    X_v, y_v = make_blobs(n_samples=300, centers=4, random_state=42, cluster_std=1.2)
    
    t0 = time.time()
    ova = ParallelOneVsAllClassifier(BinaryLinearSVM(n_iters=300), n_jobs=-1).fit(X_v, y_v)
    ovo = ParallelOneVsOneClassifier(BinaryLinearSVM(n_iters=300), n_jobs=-1).fit(X_v, y_v)
    ecoc = ECOCClassifier(BinaryLinearSVM(n_iters=300), code_size=15).fit(X_v, y_v)
    print(f"  → Huấn luyện xong 3 mô hình ranh giới trong {time.time() - t0:.2f}s")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x_min, x_max = X_v[:, 0].min() - 1, X_v[:, 0].max() + 1
    y_min, y_max = X_v[:, 1].min() - 1, X_v[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    
    for ax, clf, title in zip(axes, [ova, ovo, ecoc], ["OVA (Parallel)", "OVO (Parallel)", "ECOC"]):
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        ax.scatter(X_v[:, 0], X_v[:, 1], c=y_v, edgecolors='k', cmap='RdYlBu', s=30)
        ax.set_title(f"{title} Boundary", fontweight='bold')
        
    plt.savefig(os.path.join(FIG_DIR, 'optimized_boundaries.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # PHẦN 2: PHÂN TÍCH LỖI OVO CHUYÊN SÂU (PAIRWISE ERRORS)
    # ---------------------------------------------------------
    print("\n[2/5] Đang phân tích OVO chuyên sâu (Pairwise Confusion Matrix)...")
    X, y = make_classification(n_samples=800, n_features=10, n_informative=8, 
                               n_classes=5, n_clusters_per_class=1, random_state=42)
    X_train, X_test = X[:600], X[600:]
    y_train, y_test = y[:600], y[600:]

    ovo_opt = ParallelOneVsOneClassifier(base_estimator=BinaryLinearSVM(n_iters=500), n_jobs=-1).fit(X_train, y_train)
    conf_matrix = ovo_opt.pairwise_confusion_matrix(X_test, y_test)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=ovo_opt.classes_, yticklabels=ovo_opt.classes_)
    plt.title("Parallel OVO Pairwise Confusion Matrix", fontweight='bold')
    plt.xlabel("Dự đoán")
    plt.ylabel("Thực tế")
    plt.savefig(os.path.join(FIG_DIR, 'optimized_ovo_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # PHẦN 3: BẢNG SO SÁNH HIỆU NĂNG TỔNG HỢP
    # ---------------------------------------------------------
    print("\n[3/5] Đang huấn luyện hệ thống so sánh hiệu năng tổng hợp (Bảng Accuracy & ECOC)...")
    X_p, y_p = make_classification(n_samples=1200, n_features=25, n_informative=20, n_classes=5, class_sep=0.8, random_state=42)
    X_tr, X_te = X_p[:900], X_p[900:]
    y_tr, y_te = y_p[:900], y_p[900:]

    base_clf = BinaryLinearSVM(n_iters=500)
    models = {
        "Parallel OVA": ParallelOneVsAllClassifier(base_estimator=base_clf, n_jobs=-1),
        "Parallel OVO": ParallelOneVsOneClassifier(base_estimator=base_clf, n_jobs=-1),
        "Fast Tree": FastDecisionTreeClassifier(max_depth=5),
        "ECOC (Soft)": ECOCClassifier(base_estimator=base_clf, code_size=20, use_confidence=True)
    }

    results = []
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_tr, y_tr)
        fit_time = time.time() - t0
        acc = accuracy_score(y_te, model.predict(X_te))
        d, r0 = "-", "-"
        if hasattr(model, 'get_min_hamming_distance'):
            d, r0 = model.get_min_hamming_distance()
        results.append({"Chiến lược": name, "Accuracy": acc, "Thời gian (s)": round(fit_time, 4), "Hamming (d)": d, "Sửa lỗi (r0)": r0})

    df_res = pd.DataFrame(results)
    print("\nBảng kết quả hiệu năng (Phiên bản Tối ưu hóa):")
    print(df_res.to_string(index=False))

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x="Accuracy", y="Chiến lược", data=df_res, palette="viridis")
    plt.title("SO SÁNH ĐỘ CHÍNH XÁC CÁC MÔ HÌNH TỐI ƯU", fontweight='bold')
    plt.xlim(df_res["Accuracy"].min() - 0.05, 1.0)
    for i, v in enumerate(df_res["Accuracy"]):
        ax.text(v + 0.005, i, f"{v:.4f}", va='center', fontweight='bold')
    plt.savefig(os.path.join(FIG_DIR, 'optimized_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # PHẦN 4: HỘI TỤ ADABOOST.MH TỐI ƯU BỘ NHỚ (KHÔNG DÙNG 999999.0)
    # ---------------------------------------------------------
    print("\n[4/5] Đang kiểm chứng sự hội tụ của AdaBoost.MH Tối ưu bộ nhớ (Memory-efficient)...")
    X_b, y_b = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42)
    Y_ml = to_multilabel(y_b, k=3)
    
    F_vals, err_vals = [], []
    t0 = time.time()
    for t in range(1, 41):
        model_t = MemoryEfficientAdaBoostMH(T=t)
        model_t.fit(X_b, Y_ml)
        scores = model_t.predict_scores(X_b, k=3)
        F_vals.append(np.sum(np.exp(-Y_ml * scores)))
        err_vals.append(np.mean(model_t.predict(X_b, k=3) != y_b))
    print(f"  → Huấn luyện xong 40 mô hình AdaBoost.MH trong {time.time() - t0:.2f}s (Cực nhanh, không tốn RAM!)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(range(1, 41), F_vals, color='tab:orange', linewidth=2)
    ax1.set_xlabel('Số vòng lặp $T$')
    ax1.set_ylabel('$F(\\alpha)$')
    ax1.set_title('Hội tụ hàm mục tiêu AdaBoost.MH (Memory-efficient)')
    ax1.grid(alpha=0.3)

    ax2.plot(range(1, 41), err_vals, color='tab:red', linewidth=2)
    ax2.set_xlabel('Số vòng lặp $T$')
    ax2.set_ylabel('Training error')
    ax2.set_title('Training error giảm dần')
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'optimized_adaboost_convergence.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # PHẦN 5: KIỂM ĐỊNH T-TEST (10 RUNS)
    # ---------------------------------------------------------
    print("\n[5/5] Đang chạy 10 thực nghiệm độc lập cho T-test giữa Parallel OVA và Parallel OVO...")
    ova_accs, ovo_accs = [], []
    for i in range(10):
        X_h, y_h = make_classification(n_samples=500, n_features=15, n_informative=10, n_classes=3, random_state=i)
        X_tr, X_te = X_h[:350], X_h[350:]
        y_tr, y_te = y_h[:350], y_h[350:]
        ova_accs.append(accuracy_score(y_te, ParallelOneVsAllClassifier(BinaryLinearSVM(n_iters=200), n_jobs=-1).fit(X_tr, y_tr).predict(X_te)))
        ovo_accs.append(accuracy_score(y_te, ParallelOneVsOneClassifier(BinaryLinearSVM(n_iters=200), n_jobs=-1).fit(X_tr, y_tr).predict(X_te)))

    t_stat, p_val = ttest_rel(ova_accs, ovo_accs)
    print(f"KẾT QUẢ T-TEST: p-value = {p_val:.4f}")
    if p_val < 0.05: print(" -> Kết luận: Sự khác biệt có ý nghĩa thống kê.")
    else: print(" -> Kết luận: Khác biệt không đáng kể.")

    plt.figure(figsize=(7, 5))
    plt.boxplot([ova_accs, ovo_accs], labels=['Parallel OVA', 'Parallel OVO'], patch_artist=True)
    plt.title("PHÂN PHỐI ACCURACY (10 RUNS - PARALLEL MODELS)", fontweight='bold')
    plt.ylabel("Accuracy")
    plt.xlabel("Mô hình")
    plt.savefig(os.path.join(FIG_DIR, 'optimized_ttest.png'), dpi=300)
    plt.close()

    print(f"\n HOÀN TẤT BỘ PHÂN TÍCH ĐỈNH CAO TRONG {time.time() - start_total:.2f} GIÂY.")
    print(f" Toàn bộ biểu đồ phân tích tối ưu đã được lưu tại: {FIG_DIR}\n")

if __name__ == "__main__":
    run_optimized_rich_analysis()
