import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_classification, load_digits, fetch_openml, load_svmlight_file
from sklearn.model_selection import train_test_split
import scipy.sparse as sp

def load_synthetic_data(seed=42):
    """Sinh dữ liệu tổng hợp 2D (Blob & Classification)"""
    np.random.seed(seed)
    X_blob, y_blob = make_blobs(n_samples=300, centers=3, n_features=2, random_state=seed)
    X_tr_blob, X_te_blob, y_tr_blob, y_te_blob = train_test_split(
        X_blob, y_blob, test_size=0.2, random_state=seed)
        
    X_cls, y_cls = make_classification(n_samples=150, n_classes=3, n_features=2, 
                                       n_informative=2, n_redundant=0, 
                                       n_clusters_per_class=1, random_state=seed)
    X_tr_cls, X_te_cls, y_tr_cls, y_te_cls = train_test_split(
        X_cls, y_cls, test_size=0.2, random_state=seed)
        
    return (X_tr_blob, X_te_blob, y_tr_blob, y_te_blob), (X_tr_cls, X_te_cls, y_tr_cls, y_te_cls)

def load_mnist_data(subsample=1000, test_size=0.2, seed=42):
    """
    Nạp bộ dữ liệu Fashion-MNIST từ file vật lý data/FashionMNIST/fashion-mnist_train.csv.
    Sử dụng subsample để huấn luyện nhanh và không bị tràn bộ nhớ.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    csv_path = os.path.join(base_dir, 'data', 'FashionMNIST', 'fashion-mnist_train.csv')
    
    if os.path.exists(csv_path):
        print(f" Đang nạp bộ dữ liệu Fashion-MNIST từ file vật lý: {csv_path}...")
        import pandas as pd
        df = pd.read_csv(csv_path)
        y = df.iloc[:, 0].values
        X = df.iloc[:, 1:].values / 255.0 
        
        if subsample and subsample < len(X):
            np.random.seed(seed)
            indices = np.random.choice(len(X), subsample, replace=False)
            X, y = X[indices], y[indices]
            
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
        print(f"  → Đã nạp thành công Fashion-MNIST vật lý: Train={X_tr.shape}, Test={X_te.shape}\n")
        return X_tr, X_te, y_tr, y_te
    else:
        print(" Đang nạp bộ dữ liệu MNIST (load_digits 8x8 - 10 lớp)...")
        digits = load_digits()
        X, y = digits.data, digits.target
        X = X / 16.0 
        
        if subsample and subsample < len(X):
            np.random.seed(seed)
            indices = np.random.choice(len(X), subsample, replace=False)
            X, y = X[indices], y[indices]
            
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
        print(f"  → Đã nạp thành công: Train={X_tr.shape}, Test={X_te.shape}\n")
        return X_tr, X_te, y_tr, y_te

def load_eurlex_data(file_path=None, n_samples=1000, n_features=200, n_classes=10, seed=42):
    """
    Nạp bộ dữ liệu EUR-Lex (Ma trận thưa TF-IDF, Multi-label / Mất cân bằng).
    Nếu có file vật lý, trích xuất tập con (subsample) để tránh tràn 23GB RAM.
    """
    print(" Đang nạp bộ dữ liệu EUR-Lex (Extreme Multi-label / Ma trận thưa)...")
    
    if file_path and os.path.exists(file_path):
        print(f"  → Đang đọc file LibSVM vật lý từ: {file_path}")
        with open(file_path, 'rb') as f:
            f.readline() 
            X, y = load_svmlight_file(f, multilabel=True)
        
        n_s = min(n_samples, X.shape[0])
        n_f = min(n_features, X.shape[1])
        X_sub = X[:n_s, :n_f]
        y_sub = y[:n_s]
        
        X_dense = X_sub.toarray()
        
        Y_bin = np.zeros((n_s, n_classes), dtype=int)
        for i, labels in enumerate(y_sub):
            for lbl in labels:
                if int(lbl) < n_classes:
                    Y_bin[i, int(lbl)] = 1
                
        X_tr, X_te, Y_tr, Y_te = train_test_split(X_dense, Y_bin, test_size=0.2, random_state=seed)
        print(f"  → Đã nạp thành công EUR-Lex vật lý: Train={X_tr.shape}, Test={X_te.shape}, Nhãn={n_classes} lớp\n")
        return X_tr, X_te, Y_tr, Y_te
        
    print("  → (Chưa có file vật lý) Đang tự động mô phỏng ma trận thưa chuẩn EUR-Lex (TF-IDF)...")
    np.random.seed(seed)
    
    X = sp.random(n_samples, n_features, density=0.05, format='csr', random_state=seed)
    X.data = np.round(np.random.uniform(0.01, 0.8, size=X.data.shape), 4)
    
    class_probs = np.exp(-np.linspace(0, 3, n_classes))
    class_probs /= class_probs.sum()
    
    y_list = []
    for _ in range(n_samples):
        n_labels = np.random.poisson(lam=3) + 1
        n_labels = min(n_labels, n_classes)
        labels = np.random.choice(n_classes, size=n_labels, replace=False, p=class_probs)
        y_list.append(labels)
        
    Y_bin = np.zeros((n_samples, n_classes), dtype=int)
    for i, labels in enumerate(y_list):
        Y_bin[i, labels] = 1
        
    X_tr, X_te, Y_tr, Y_te = train_test_split(X.toarray(), Y_bin, test_size=0.2, random_state=seed)
    print(f"  → Đã mô phỏng thành công EUR-Lex: Train={X_tr.shape}, Test={X_te.shape}, Nhãn={n_classes} lớp\n")
    return X_tr, X_te, Y_tr, Y_te
