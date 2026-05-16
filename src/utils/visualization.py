import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_classification
from sklearn.model_selection import train_test_split

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