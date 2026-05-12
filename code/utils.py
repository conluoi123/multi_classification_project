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

def plot_boundary(clf, X, y, title, ax):
    """Vẽ ranh giới quyết định"""
    h = 0.05
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Gom lưới điểm để đưa vào hàm predict
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    try:
        Z = clf.predict(grid_points)
    except:
        # Trường hợp predict nhận từng điểm một
        Z = np.array([clf.predict(pt.reshape(1, -1))[0] for pt in grid_points])
        
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='tab10')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10', edgecolors='k', s=25)
    ax.set_title(title)