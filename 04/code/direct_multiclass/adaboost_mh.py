import numpy as np
from src.base.decision_stump import DecisionStump

def to_multilabel(y, k):
    """Chuyển nhãn đơn sang ma trận nhãn {-1, +1}^k."""
    m = len(y)
    Y = -np.ones((m, k), dtype=float)
    Y[np.arange(m), y] = 1.0
    return Y

class AdaBoostMH:
    """AdaBoost.MH cài từ đầu"""
    def __init__(self, T=50):
        self.T = T

    def get_expanded_X(self, X, k):
        """Tạo đặc trưng riêng cho từng lớp"""
        m, n = X.shape
        X_exp = np.repeat(X, k, axis=0)
        
        # Độn ma trận bằng số cực lớn để Stump chỉ cắt trên đúng 1 lớp
        X_isolated = np.full((m * k, n * k), 999999.0)
        for l in range(k):
            row_indices = np.arange(m) * k + l
            X_isolated[row_indices, l*n : (l+1)*n] = X
            
        return np.hstack([X_exp, X_isolated])

    def fit(self, X, y_multilabel):
        m, k = y_multilabel.shape
        X_exp_full = self.get_expanded_X(X, k)
        y_exp = y_multilabel.flatten()
        
        D = np.ones(m * k) / (m * k)
        self.stumps, self.alphas = [], []
        
        for _ in range(self.T):
            stump = DecisionStump()
            stump.fit(X_exp_full, y_exp, D)
            pred = stump.predict(X_exp_full)
            
            eps = np.sum(D[pred != y_exp])
            eps = np.clip(eps, 1e-10, 1 - 1e-10)
            
            # Nếu vi phạm điều kiện học yếu, dừng thuật toán sớm
            if eps >= 0.5:
                break 
                
            alpha = 0.5 * np.log((1 - eps) / eps)
            D *= np.exp(-alpha * y_exp * pred)
            D /= D.sum()
            
            self.stumps.append(stump)
            self.alphas.append(alpha)

    def predict_scores(self, X, k):
        m = len(X)
        scores = np.zeros((m, k))
        X_exp_full = self.get_expanded_X(X, k)
        
        for alpha, stump in zip(self.alphas, self.stumps):
            p_pred = stump.predict(X_exp_full)
            scores += alpha * p_pred.reshape(m, k)
        return scores

    def predict(self, X, k):
        scores = self.predict_scores(X, k)
        return np.argmax(scores, axis=1)