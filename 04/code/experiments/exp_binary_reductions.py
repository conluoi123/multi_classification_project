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
FIG_DIR = os.path.join(BASE_DIR, 'figures', 'binary_reductions')
os.makedirs(FIG_DIR, exist_ok=True)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.base.binary_svm import BinaryLinearSVM
from src.binary_reductions.ova_classifier import OneVsAllClassifier, CalibratedOneVsAllClassifier
from src.binary_reductions.ovo_classifier import OneVsOneClassifier
from src.binary_reductions.ecoc_classifier import ECOCClassifier

def run_visual_experiments():
    print("PHẦN 1: TRỰC QUAN HÓA MA TRẬN (OVA, OVO, ECOC) & RANH GIỚI QUYẾT ĐỊNH")
    print("="*65)
    k = 4
    
    M_ova = np.eye(k) * 2 - 1
    M_ovo = np.zeros((k, k*(k-1)//2))
    col = 0
    for i in range(k):
        for j in range(i+1, k):
            M_ovo[i, col], M_ovo[j, col] = 1, -1
            col += 1
    M_ecoc = np.random.choice([-1, 1], size=(k, 12)) 

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cmap = sns.color_palette("coolwarm", 3)
    
    sns.heatmap(M_ova, annot=True, cmap=cmap, cbar=False, ax=axes[0], linewidths=1, linecolor='black')
    axes[0].set_title("Ma trận OVA (KxK)", fontweight='bold')
    
    sns.heatmap(M_ovo, annot=True, cmap=cmap, cbar=False, ax=axes[1], linewidths=1, linecolor='black')
    axes[1].set_title(f"Ma trận OVO (K x {int(k*(k-1)/2)})", fontweight='bold')
    
    sns.heatmap(M_ecoc, annot=True, cmap=cmap, cbar=False, ax=axes[2], linewidths=1, linecolor='black')
    axes[2].set_title("Ma trận ECOC Random (K x c=12)", fontweight='bold')
    
    plt.savefig(os.path.join(FIG_DIR, 'all_coding_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()

    X_v, y_v = make_blobs(n_samples=300, centers=k, random_state=42, cluster_std=1.2)
    print("Đang huấn luyện mô hình để vẽ Ranh giới quyết định...")
    
    ova = OneVsAllClassifier(BinaryLinearSVM(n_iters=300)).fit(X_v, y_v)
    ovo = OneVsOneClassifier(BinaryLinearSVM(n_iters=300)).fit(X_v, y_v)
    ecoc = ECOCClassifier(BinaryLinearSVM(n_iters=300), code_size=15).fit(X_v, y_v)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x_min, x_max = X_v[:, 0].min() - 1, X_v[:, 0].max() + 1
    y_min, y_max = X_v[:, 1].min() - 1, X_v[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    
    for ax, clf, title in zip(axes, [ova, ovo, ecoc], ["OVA", "OVO", "ECOC"]):
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        ax.scatter(X_v[:, 0], X_v[:, 1], c=y_v, edgecolors='k', cmap='RdYlBu', s=30)
        ax.set_title(f"{title} Boundary", fontweight='bold')
        
    plt.savefig(os.path.join(FIG_DIR, 'decision_boundaries_all.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Đã lưu Heatmaps và Boundaries vào {FIG_DIR}\n")

def run_ovo_detailed_analysis():
    print("PHẦN 2: PHÂN TÍCH OVO CHUYÊN SÂU (PAIRWISE ERRORS)")
    print("="*65)
    X, y = make_classification(n_samples=800, n_features=10, n_informative=8, 
                               n_classes=5, n_clusters_per_class=1, random_state=42)
    X_train, X_test = X[:600], X[600:]
    y_train, y_test = y[:600], y[600:]

    ovo = OneVsOneClassifier(base_estimator=BinaryLinearSVM(n_iters=500)).fit(X_train, y_train)
    
    conf_matrix = ovo.pairwise_confusion_matrix(X_test, y_test)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt=".2f", cmap="YlGnBu", 
                xticklabels=ovo.classes_, yticklabels=ovo.classes_)
    plt.title("OVO Pairwise Confusion Matrix (Error Analysis)", fontweight='bold')
    plt.xlabel("Dự đoán")
    plt.ylabel("Thực tế")
    
    plt.savefig(os.path.join(FIG_DIR, 'ovo_pairwise_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    try:
        clf_pair = ovo.get_pairwise_classifier(class_a=0, class_b=1)
        print(f"Truy xuất bộ phân loại (0 vs 1) thành công. Bias b = {clf_pair.b:.4f}")
    except: pass
    print(f"Đã lưu Ma trận nhầm lẫn OVO.\n")

def run_performance_experiments():
    print("PHẦN 3: SO SÁNH HIỆU NĂNG & SỨC MẠNH ECOC")
    print("="*65)
    
    X, y = make_classification(n_samples=1200, n_features=25, n_informative=20, 
                               n_classes=5, class_sep=0.8, random_state=42)
    X_train, X_test = X[:900], X[900:]
    y_train, y_test = y[:900], y[900:]

    base_clf = BinaryLinearSVM(n_iters=500)
    models = {
        "OVA": OneVsAllClassifier(base_estimator=base_clf),
        "OVO": OneVsOneClassifier(base_estimator=base_clf),
        "ECOC (Hard)": ECOCClassifier(base_estimator=base_clf, code_size=20, use_confidence=False),
        "ECOC (Soft)": ECOCClassifier(base_estimator=base_clf, code_size=20, use_confidence=True)
    }

    results = []
    print("Đang huấn luyện hệ thống đánh giá...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        d, r0 = "-", "-"
        if hasattr(model, 'get_min_hamming_distance'):
            d, r0 = model.get_min_hamming_distance()
        results.append({"Chiến lược": name, "Accuracy": acc, "Hamming (d)": d, "Sửa lỗi (r0)": r0})

    df_res = pd.DataFrame(results)
    print("\nBảng kết quả hiệu năng:")
    print(df_res.to_string(index=False))

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x="Accuracy", y="Chiến lược", data=df_res, palette="viridis")
    plt.title("SO SÁNH ĐỘ CHÍNH XÁC (ACCURACY)", fontweight='bold')
    plt.xlim(df_res["Accuracy"].min() - 0.05, 1.0)
    for i, v in enumerate(df_res["Accuracy"]):
        ax.text(v + 0.005, i, f"{v:.4f}", va='center', fontweight='bold')
    
    plt.savefig(os.path.join(FIG_DIR, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    c_lengths = [5, 10, 20, 40, 60]
    acc_s = []
    for c in c_lengths:
        m = ECOCClassifier(base_clf, code_size=c, use_confidence=True).fit(X_train, y_train)
        acc_s.append(accuracy_score(y_test, m.predict(X_test)))
    
    plt.figure(figsize=(9, 4))
    plt.plot(c_lengths, acc_s, 's-', linewidth=2, color='blue', label='ECOC Soft Decoding')
    plt.xlabel("Chiều dài mã (c)")
    plt.ylabel("Accuracy")
    plt.title("ECOC: TĂNG CHIỀU DÀI MÃ GIÚP TĂNG KHẢ NĂNG SỬA LỖI", fontweight='bold')
    plt.xticks(c_lengths)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig(os.path.join(FIG_DIR, 'ecoc_depth_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Đã lưu kết quả Hiệu năng và ECOC Depth.\n")

def run_calibration_experiments():
    print("PHẦN 4: KIỂM CHỨNG VẤN ĐỀ HIỆU CHUẨN (CALIBRATION ISSUE)")
    print("="*65)
    X_imb, y_imb = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                                       n_classes=4, weights=[0.05, 0.1, 0.25, 0.6], random_state=42)
    X_tr, X_te = X_imb[:700], X_imb[700:]
    y_tr, y_te = y_imb[:700], y_imb[700:]

    raw = OneVsAllClassifier(BinaryLinearSVM(n_iters=300)).fit(X_tr, y_tr)
    cal = CalibratedOneVsAllClassifier(BinaryLinearSVM(n_iters=300)).fit(X_tr, y_tr)

    acc_r = accuracy_score(y_te, raw.predict(X_te))
    acc_c = accuracy_score(y_te, cal.predict(X_te))

    plt.figure(figsize=(6, 5))
    sns.barplot(x=['OVA Thô', 'OVA Calibrated'], y=[acc_r, acc_c], palette="coolwarm")
    plt.title("HIỆU QUẢ CỦA PLATT SCALING TRÊN DỮ LIỆU MẤT CÂN BẰNG", fontweight='bold')
    plt.ylabel("Độ chính xác")
    plt.xlabel("Mô hình")
    plt.ylim(min(acc_r, acc_c) - 0.05, 1.0)
    for i, v in enumerate([acc_r, acc_c]):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center', fontweight='bold')

    plt.savefig(os.path.join(FIG_DIR, 'calibration_impact.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f" - OVA Thô: {acc_r:.4f} | OVA Calibrated: {acc_c:.4f}")
    print(f"Đã lưu ảnh so sánh Calibration.\n")

def run_complexity_and_stats():
    print("PHẦN 5: ĐỘ PHỨC TẠP (TABLE 8.1) & KIỂM ĐỊNH T-TEST")
    print("="*65)
    
    k_vals = [3, 5, 8, 12, 15]
    t_ova, t_ovo = [], []
    for k in k_vals:
        Xt, yt = make_classification(n_samples=150*k, n_features=10, n_classes=k, n_informative=8, random_state=42)
        t0 = time.time()
        OneVsAllClassifier(BinaryLinearSVM(n_iters=200)).fit(Xt, yt)
        t_ova.append(time.time() - t0)
        t0 = time.time()
        OneVsOneClassifier(BinaryLinearSVM(n_iters=200)).fit(Xt, yt)
        t_ovo.append(time.time() - t0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(k_vals, t_ova, 'o-', label='OVA Time', color='#d62728', linewidth=2)
    ax.plot(k_vals, t_ovo, 's-', label='OVO Time', color='#1f77b4', linewidth=2)
    ax.set_xticks(k_vals)
    ax.set_xlabel("Số lượng lớp (k)")
    ax.set_ylabel("Thời gian (s)")
    ax.set_title("KIỂM CHỨNG ĐỘ PHỨC TẠP THỜI GIAN (TABLE 8.1 FML)", fontweight='bold')
    ax.legend()
    plt.savefig(os.path.join(FIG_DIR, 'complexity_time.png'), dpi=300, bbox_inches='tight')
    plt.close()

    ova_accs, ovo_accs = [], []
    print("Đang chạy 10 thực nghiệm độc lập cho T-test...")
    for i in range(10):
        X_h, y_h = make_classification(n_samples=500, n_features=15, n_informative=10, n_classes=3, random_state=i)
        X_tr, X_te = X_h[:350], X_h[350:]
        y_tr, y_te = y_h[:350], y_h[350:]
        ova_accs.append(accuracy_score(y_te, OneVsAllClassifier(BinaryLinearSVM(n_iters=200)).fit(X_tr, y_tr).predict(X_te)))
        ovo_accs.append(accuracy_score(y_te, OneVsOneClassifier(BinaryLinearSVM(n_iters=200)).fit(X_tr, y_tr).predict(X_te)))

    t_stat, p_val = ttest_rel(ova_accs, ovo_accs)
    print(f"KẾT QUẢ T-TEST: p-value = {p_val:.4f}")
    if p_val < 0.05: print(" -> Kết luận: Sự khác biệt có ý nghĩa thống kê.")
    else: print(" -> Kết luận: Khác biệt không đáng kể.")

    plt.figure(figsize=(7, 5))
    plt.boxplot([ova_accs, ovo_accs], labels=['OVA', 'OVO'], patch_artist=True)
    plt.title("PHÂN PHỐI ACCURACY (10 RUNS)", fontweight='bold')
    plt.ylabel("Accuracy")
    plt.xlabel("Mô hình")
    plt.savefig(os.path.join(FIG_DIR, 'hypothesis_test.png'), dpi=300)
    plt.close()
    print(f"Đã lưu kết quả Complexity và T-test.\n")

if __name__ == "__main__":
    start_total = time.time()
    run_visual_experiments()
    run_ovo_detailed_analysis()
    run_performance_experiments()
    run_calibration_experiments()
    run_complexity_and_stats()
    print(f"HOÀN THÀNH TOÀN BỘ THỰC NGHIỆM TRONG {time.time() - start_total:.2f} GIÂY.")
