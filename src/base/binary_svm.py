import numpy as np

class BinaryLinearSVM:
    """
    Bộ phân loại nhị phân Linear SVM cài đặt bằng Vectorized Gradient Descent.
    Tối ưu hóa hàm mục tiêu: L = lambda ||w||^2 + (1/N) * sum(max(0, 1 - y_i(w^T x_i + b)))
    """
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iters=1000, tol=1e-4):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.tol = tol 
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        
        for _ in range(self.n_iters):
            w_prev = np.copy(self.w)
            
            margins = y * (np.dot(X, self.w) + self.b)
            
            misclassified_idx = np.where(margins < 1)[0]
            
            dw = 2 * self.lambda_param * self.w
            db = 0
            
            if len(misclassified_idx) > 0:
                dw -= np.dot(X[misclassified_idx].T, y[misclassified_idx]) / n_samples
                db -= np.sum(y[misclassified_idx]) / n_samples
                
            self.w -= self.lr * dw
            self.b -= self.lr * db
            
            if np.linalg.norm(self.w - w_prev) < self.tol:
                break
                
        return self

    def decision_function(self, X):
        """Trả về khoảng cách có hướng tới siêu phẳng (Margin)"""
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))