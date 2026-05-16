import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_classification
from sklearn.model_selection import train_test_split

def load_synthetic_data(seed=42):
    """Sinh dữ liệu tổng hợp"""
    np.random.seed(seed)
    
    # Dữ liệu blob 3 lớp, 2 chiều (Dùng cho Decision Tree và AdaBoost.MH)
    X_blob, y_blob = make_blobs(n_samples=300, centers=3, n_features=2, random_state=seed)
    X_tr_blob, X_te_blob, y_tr_blob, y_te_blob = train_test_split(
        X_blob, y_blob, test_size=0.2, random_state=seed)
    # Dữ liệu phân loại tổng quát 3 lớp, 2 chiều (Dùng cho Multi-class SVM)
    X_cls, y_cls = make_classification(n_samples=150, n_classes=3, n_features=2, 
                                       n_informative=2, n_redundant=0, 
                                       n_clusters_per_class=1, random_state=seed)
    X_tr_cls, X_te_cls, y_tr_cls, y_te_cls = train_test_split(
        X_cls, y_cls, test_size=0.2, random_state=seed)
        
    return (X_tr_blob, X_te_blob, y_tr_blob, y_te_blob), (X_tr_cls, X_te_cls, y_tr_cls, y_te_cls)
