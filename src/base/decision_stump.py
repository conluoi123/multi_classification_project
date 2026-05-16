import numpy as np 

class DecisionStump:
    """Cây sâu 1 - base learner cho AdaBoost.MH."""
    def fit(self, X, y, w):
        m, n = X.shape
        self.j, self.t, self.p = None, None, 1
        best_err = np.inf
        for j in range(n):
            for t in np.unique(X[:, j]):
                # Bỏ qua các giá trị độn 999999.0 để tăng tốc
                if t >= 999999.0: continue 
                for p in [1, -1]:
                    pred = np.where(X[:, j] <= t, p, -p)
                    err = np.sum(w[pred != y])
                    if err < best_err:
                        best_err = err
                        self.j, self.t, self.p = j, t, p

    def predict(self, X):
        return np.where(X[:, self.j] <= self.t, self.p, -self.p)
