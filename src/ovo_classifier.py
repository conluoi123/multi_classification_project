import numpy as np
from typing import Any, Optional, List, Tuple, Dict
from copy import deepcopy
from itertools import combinations
import warnings


class OneVsOneClassifier:
    """
    Chiến lược phân loại đa lớp One-Vs-One (Một-đối-Một).
    """
    def __init__(
        self,
        base_estimator,
        n_jobs: int = 1,
        verbose: bool = False
    ):
        self.base_estimator = base_estimator
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        self.classifiers: Dict[Tuple[int, int], Any] = {}
        self.n_classes: Optional[int] = None
        self.classes_: Optional[np.ndarray] = None
        self.pairwise_indices_: List[Tuple[int, int]] = []
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Huấn luyện bộ phân loại OVO."""
        if len(X) != len(y):
            raise ValueError(f"X và y phải có cùng độ dài: {len(X)} so với {len(y)}")

        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        
        if self.n_classes < 2:
            raise ValueError(f"Cần ít nhất 2 lớp, hiện có {self.n_classes}")

        self.pairwise_indices_ = list(combinations(range(self.n_classes), 2))
        n_classifiers = len(self.pairwise_indices_)
        
        if self.verbose:
            print(f"Đang huấn luyện OVO với {self.n_classes} lớp...")
            print(f"Số lượng bộ phân loại nhị phân: {n_classifiers}")

        self.classifiers = {}
        
        for idx, (i, j) in enumerate(self.pairwise_indices_):
            class_i = self.classes_[i]
            class_j = self.classes_[j]
            
            if self.verbose:
                print(f"  [{idx+1}/{n_classifiers}] Đang huấn luyện bộ phân loại cho cặp lớp {class_i} vs {class_j}...")

            mask = (y == class_i) | (y == class_j)
            X_pair = X[mask]
            y_pair = y[mask]

            y_binary = np.where(y_pair == class_i, -1, 1)
            
            if self.verbose and len(X_pair) > 0:
                n_class_i = np.sum(y_pair == class_i)
                n_class_j = np.sum(y_pair == class_j)
                print(f"      Huấn luyện trên {len(X_pair)} mẫu "
                      f"({n_class_i} từ lớp {class_i}, "
                      f"{n_class_j} từ lớp {class_j})")
            
            classifier = deepcopy(self.base_estimator)
            
            try:
                classifier.fit(X_pair, y_binary)
                self.classifiers[(i, j)] = classifier
            except Exception as e:
                warnings.warn(f"Lỗi khi huấn luyện bộ phân loại cho cặp ({i},{j}): {e}")
                self.classifiers[(i, j)] = DummyClassifier(prediction=-1)
        
        if self.verbose:
            print("Quá trình huấn luyện OVO hoàn tất.")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Dự đoán lớp bằng phương pháp bỏ phiếu đa số (Majority Voting), 
        có áp dụng xử lý hòa (Tie-breaking) dựa trên độ tin cậy.
        """
        if not self.classifiers:
            raise ValueError("Bộ phân loại chưa được huấn luyện. Hãy gọi fit() trước.")
        
        n_test = len(X)
        
        votes = np.zeros((n_test, self.n_classes), dtype=int)
        
        tie_breaker_scores = np.zeros((n_test, self.n_classes))
        
        for (i, j), classifier in self.classifiers.items():
            predictions = classifier.predict(X)
            votes[:, i] += (predictions == -1).astype(int)
            votes[:, j] += (predictions == 1).astype(int)
            
            if hasattr(classifier, 'decision_function'):
                dec_vals = classifier.decision_function(X)
            else:
                dec_vals = predictions.astype(float)
            
            tie_breaker_scores[:, i] -= dec_vals
            tie_breaker_scores[:, j] += dec_vals
            
        y_pred = []
        for sample_idx in range(n_test):
            max_vote = np.max(votes[sample_idx])
            candidates = np.where(votes[sample_idx] == max_vote)[0]
            
            if len(candidates) == 1:
                best_idx = candidates[0]
            else:
                best_idx = candidates[np.argmax(tie_breaker_scores[sample_idx, candidates])]
                
            y_pred.append(self.classes_[best_idx])
            
        return np.array(y_pred)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Tính toán bảng đếm phiếu thô cho mỗi lớp."""
        n_test = len(X)
        scores = np.zeros((n_test, self.n_classes))
        
        for (i, j), classifier in self.classifiers.items():
            predictions = classifier.predict(X)
            scores[:, i] += (predictions == -1).astype(float)
            scores[:, j] += (predictions == 1).astype(float)
        
        return scores
    
    def get_pairwise_classifier(self, class_a, class_b):
        """Truy xuất bộ phân loại nhị phân đã huấn luyện cho cặp lớp cụ thể."""
        idx_a = np.where(self.classes_ == class_a)[0]
        idx_b = np.where(self.classes_ == class_b)[0]
        
        if len(idx_a) == 0 or len(idx_b) == 0:
            raise ValueError(f"Không tìm thấy lớp {class_a} hoặc {class_b}")
        
        i, j = idx_a[0], idx_b[0]
        
        if i > j:
            i, j = j, i
        
        if (i, j) not in self.classifiers:
            raise ValueError(f"Không tìm thấy bộ phân loại cho cặp ({class_a}, {class_b})")
        
        return self.classifiers[(i, j)]
    
    def pairwise_confusion_matrix(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Tính toán ma trận tỷ lệ dự đoán giữa các cặp lớp để phân tích sai số."""
        n_classes = self.n_classes
        confusion = np.zeros((n_classes, n_classes))
        
        y_pred = self.predict(X)
        
        for i in range(n_classes):
            class_i = self.classes_[i]
            mask_i = (y == class_i)
            
            if mask_i.sum() == 0:
                continue
            
            for j in range(n_classes):
                class_j = self.classes_[j]
                confusion[i, j] = np.mean(y_pred[mask_i] == class_j)
        
        return confusion


class DummyClassifier:
    """Bộ phân loại giả định luôn dự đoán một giá trị cố định."""
    def __init__(self, prediction=-1):
        self.prediction = prediction
    
    def fit(self, X, y):
        return self
    
    def predict(self, X):
        return np.full(len(X), self.prediction)


class WeightedOneVsOneClassifier(OneVsOneClassifier):
    """
    Chiến lược OVO sử dụng bỏ phiếu có trọng số dựa trên điểm tin cậy (Margin).
    """
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán sử dụng cơ chế bỏ phiếu có trọng số."""
        scores = self.decision_function_weighted(X)
        winner_indices = np.argmax(scores, axis=1)
        return self.classes_[winner_indices]
    
    def decision_function_weighted(self, X: np.ndarray) -> np.ndarray:
        """Tích lũy các giá trị quyết định từ tất cả các bộ phân loại cặp."""
        n_test = len(X)
        scores = np.zeros((n_test, self.n_classes))
        
        for (i, j), classifier in self.classifiers.items():
            if hasattr(classifier, 'decision_function'):
                decision_values = classifier.decision_function(X)
            else:
                decision_values = classifier.predict(X).astype(float)
            
            scores[:, i] -= decision_values  
            scores[:, j] += decision_values  
        
        return scores