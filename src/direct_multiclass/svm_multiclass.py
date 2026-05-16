import numpy as np

class MulticlassSVM:
    """Multi-class SVM (primal, linear kernel) cài từ đầu"""
    
    def __init__(self, n_classes, lr=0.01, C=1.0, n_epochs=500):
        self.k = n_classes
        self.lr = lr
        self.C = C
        self.n_epochs = n_epochs
        self.W = None
        self.loss_history = []

    def fit(self, X, y):
        m, n = X.shape
        # Khởi tạo ma trận W có shape: (k_classes, n_features)
        self.W = np.zeros((self.k, n))
        self.loss_history = []
        
        for epoch in range(self.n_epochs):
            total_loss = 0.0
            grad = np.zeros_like(self.W)
            
            for i in range(m):
                scores = self.W @ X[i] # Lấy mảng điểm (k,)
                # Tính margins: max(0, w_l*x - w_y*x + 1)
                margins = np.maximum(0, scores - scores[y[i]] + 1.0)
                margins[y[i]] = 0.0 # Lớp đúng không tính
                
                violated = margins > 0
                
                # Cập nhật Subgradient
                grad[violated] += X[i]
                grad[y[i]] -= violated.sum() * X[i]
                total_loss += margins[violated].sum()
                
            # Cập nhật trọng số: L2 regularization + hinge loss
            self.W = (1 - self.lr) * self.W - (self.lr * self.C / m) * grad
            self.loss_history.append(total_loss / m)

    def predict(self, X):
        return np.argmax(X @ self.W.T, axis=1)