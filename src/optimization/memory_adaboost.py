from matplotlib.pylab import cond
import numpy as np
from src.base.decision_stump import DecisionStump
from src.direct_multiclass.adaboost_mh import AdaBoostMH

class MemoryEfficientDecisionStump(DecisionStump):
    """
    [TỐI ƯU HÓA BỘ NHỚ] Phiên bản Stump tối ưu hóa bộ nhớ, loại bỏ hoàn toàn ma trận độn 999999.0.
    """
    def fit(self, X, y_exp, w, k):
        m, n = X.shape
        self.j, self.t, self.p, self.l = None, None, 1, None
        best_err = np.inf
        
        W_mat = w.reshape(m, k)
        Y_mat = y_exp.reshape(m, k)
        
        for j in range(n):
            thresholds = np.unique(X[:, j])
            if len(thresholds) > 20:
                thresholds = np.quantile(X[:, j], np.linspace(0.05, 0.95, 20))
                
            for t in thresholds:
                cond = (X[:, j] <= t)
                
                # 1. Shared Stump
                pred_shared = np.where(cond[:, None], 1, -1)
                for p in [1, -1]:
                    err = np.sum(W_mat[p * pred_shared != Y_mat])
                    if err < best_err:
                        best_err, self.j, self.t, self.p, self.l = err, j, t, p, None
                        
                # 2. Class-specific Stump
                for p in [1, -1]:
                    pred_l = np.where(cond, p, -p)
                    err_l_col = np.sum(W_mat * (pred_l[:, None] != Y_mat), axis=0)
                    err_other_mat = W_mat * (-p != Y_mat)
                    total_err_other = np.sum(err_other_mat)
                    err_other_col = total_err_other - np.sum(err_other_mat, axis=0)
                    
                    errs = err_l_col + err_other_col
                    min_l = np.argmin(errs)
                    if errs[min_l] < best_err:
                        best_err, self.j, self.t, self.p, self.l = errs[min_l], j, t, p, min_l

    def predict(self, X, k):
        m = len(X)
        cond = (X[:, self.j] <= self.t)
        if self.l is None:
            pred = np.repeat(np.where(cond, self.p, -self.p)[:, None], k, axis=1)
            return pred
        else:
            pred = np.full((m, k), -self.p, dtype=float)
            pred[:, self.l] = np.where(cond, self.p, -self.p)
            return pred

class MemoryEfficientAdaBoostMH(AdaBoostMH):
    """
    [TỐI ƯU HÓA BỘ NHỚ] Phiên bản AdaBoost.MH không dùng ma trận độn 999999.0, giảm 99.75% RAM.
    """
    def fit(self, X, y_multilabel):
        m, k = y_multilabel.shape
        y_exp = y_multilabel.flatten()
        
        D = np.ones(m * k) / (m * k)
        self.stumps, self.alphas = [], []
        
        for _ in range(self.T):
            stump = MemoryEfficientDecisionStump()
            stump.fit(X, y_exp, D, k)
            pred_mat = stump.predict(X, k)
            pred_exp = pred_mat.flatten()
            
            eps = np.sum(D[pred_exp != y_exp])
            eps = np.clip(eps, 1e-10, 1 - 1e-10)
            
            if eps >= 0.5:
                break 
                
            alpha = 0.5 * np.log((1 - eps) / eps)
            D *= np.exp(-alpha * y_exp * pred_exp)
            D /= D.sum()
            
            self.stumps.append(stump)
            self.alphas.append(alpha)

    def predict_scores(self, X, k):
        m = len(X)
        scores = np.zeros((m, k))
        for alpha, stump in zip(self.alphas, self.stumps):
            p_pred = stump.predict(X, k)
            scores += alpha * p_pred
        return scores

    def predict(self, X, k):
        scores = self.predict_scores(X, k)
        return np.argmax(scores, axis=1)
