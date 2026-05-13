import numpy as np
from typing import Optional, List, Any
from copy import deepcopy
import warnings
from scipy.optimize import minimize
from scipy.special import expit, xlogy 


class OneVsAllClassifier:
    """
    Chiến lược phân loại đa lớp One-vs-All (OVA) cơ bản.
    Sử dụng điểm số Margin (f_x) nguyên thủy để quyết định lớp.
    """
    def __init__(self, base_estimator, verbose: bool = False):
        self.base_estimator = base_estimator
        self.verbose = verbose
        
        self.classifiers: List[Any] = []
        self.n_classes: Optional[int] = None
        self.classes_: Optional[np.ndarray] = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Huấn luyện k bộ phân loại nhị phân OVA."""
        if len(X) != len(y):
            raise ValueError(f"Kích thước X và y không khớp: {len(X)} vs {len(y)}")
        
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        
        if self.n_classes < 3:
            warnings.warn("OVA được thiết kế cho dữ liệu có từ 3 lớp trở lên.")
            
        self.classifiers = []
        
        for idx, class_label in enumerate(self.classes_):
            if self.verbose:
                print(f"[{idx+1}/{self.n_classes}] Đang huấn luyện mô hình cho Lớp '{class_label}' vs Phần còn lại...")
            
            y_binary = np.where(y == class_label, 1, -1)
            
            classifier = deepcopy(self.base_estimator)
            classifier.fit(X, y_binary)
            self.classifiers.append(classifier)
            
        return self
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Tính toán confidence scores f_l(x) cho tất cả các lớp.
        Có cơ chế fallback an toàn nếu mô hình không hỗ trợ decision_function.
        """
        if not self.classifiers:
            raise ValueError("Mô hình chưa được huấn luyện. Vui lòng gọi fit() trước.")
        
        n_test = len(X)
        scores = np.zeros((n_test, self.n_classes))
        
        for idx, classifier in enumerate(self.classifiers):
            if hasattr(classifier, 'decision_function'):
                scores[:, idx] = classifier.decision_function(X)
            else:
                # Fallback: dùng predict thay thế nếu không có decision_function
                scores[:, idx] = classifier.predict(X).astype(float)
            
        return scores
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Luật quyết định: h(x) = argmax_l f_l(x)"""
        scores = self.decision_function(X)
        winner_indices = np.argmax(scores, axis=1)
        return self.classes_[winner_indices]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Dự đoán xác suất bằng Softmax normalization trên điểm margin.
        """
        scores = self.decision_function(X)
        # Softmax normalization có chống tràn số (Subtract max)
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        proba = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return proba
    
    def get_classifier(self, class_label) -> Any:
        """
        Trích xuất bộ phân loại nhị phân của một lớp cụ thể.
        Dùng để debug hoặc phân tích.
        """
        class_idx = np.where(self.classes_ == class_label)[0]
        if len(class_idx) == 0:
            raise ValueError(f"Không tìm thấy lớp: {class_label}")
        return self.classifiers[class_idx[0]]


class CalibratedOneVsAllClassifier(OneVsAllClassifier):
    """
    [MỞ RỘNG] OVA được trang bị Hiệu chuẩn xác suất (Platt Scaling).
    Giải quyết bài toán "Calibration Issue" trong Mục 8.4.1 của sách FML.
    """
    def __init__(self, base_estimator, verbose: bool = False):
        super().__init__(base_estimator, verbose)
        self.calibrators: List[tuple] = [] 
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        super().fit(X, y)
        
        if self.verbose:
            print("Đang hiệu chuẩn điểm số (Platt Scaling)...")
            
        self.calibrators = []
        
        for idx, class_label in enumerate(self.classes_):
            if hasattr(self.classifiers[idx], 'decision_function'):
                f_scores = self.classifiers[idx].decision_function(X)
            else:
                f_scores = self.classifiers[idx].predict(X).astype(float)
            
            y_true_binary = np.where(y == class_label, 1, 0)
            
            # Hàm mất mát Log-loss với xlogy để chống NaN/Overflow
            def objective(params):
                A, B = params
                p = expit(A * f_scores + B) # p = 1 / (1 + exp(-(A*f + B)))
                loss = -(xlogy(y_true_binary, p) + xlogy(1 - y_true_binary, 1 - p)).mean()
                return loss
            
            # Tối ưu hóa bằng L-BFGS-B 
            res = minimize(objective, x0=[1.0, 0.0], method='L-BFGS-B')
            self.calibrators.append(tuple(res.x))
            
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán xác suất P(y=l|x) bằng hàm Sigmoid đã nắn chỉnh (Calibrated)."""
        n_test = len(X)
        probas = np.zeros((n_test, self.n_classes))
        
        for idx, (A, B) in enumerate(self.calibrators):
            if hasattr(self.classifiers[idx], 'decision_function'):
                f_scores = self.classifiers[idx].decision_function(X)
            else:
                f_scores = self.classifiers[idx].predict(X).astype(float)
                
            probas[:, idx] = expit(A * f_scores + B)
            
        epsilon = 1e-15
        row_sums = np.sum(probas, axis=1, keepdims=True)
        row_sums[row_sums == 0] = epsilon
        probas /= row_sums
        
        return probas

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Quyết định lớp dựa trên xác suất đã hiệu chuẩn."""
        probas = self.predict_proba(X)
        winner_indices = np.argmax(probas, axis=1)
        return self.classes_[winner_indices]