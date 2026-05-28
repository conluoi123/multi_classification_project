import numpy as np
import warnings
from typing import Optional, List, Any
from copy import deepcopy
from joblib import Parallel, delayed

from src.binary_reductions.ova_classifier import OneVsAllClassifier

def _fit_ova_binary_model(base_estimator, X, y, class_label, idx, n_classes, verbose):
    if verbose:
        print(f"[{idx+1}/{n_classes}] Đang huấn luyện mô hình cho Lớp '{class_label}' vs Phần còn lại (Đa luồng)...")
    y_binary = np.where(y == class_label, 1, -1)
    classifier = deepcopy(base_estimator)
    classifier.fit(X, y_binary)
    return classifier

class ParallelOneVsAllClassifier(OneVsAllClassifier):
    """
    [TỐI ƯU HÓA] Phiên bản kế thừa của OneVsAllClassifier tích hợp xử lý song song (joblib).
    Tuân thủ nguyên tắc OCP: Không sửa đổi mã gốc của thành viên khác.
    """
    def __init__(self, base_estimator, verbose: bool = False, n_jobs: int = -1):
        super().__init__(base_estimator, verbose)
        self.n_jobs = n_jobs
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Huấn luyện k bộ phân loại nhị phân OVA song song đa luồng."""
        if len(X) != len(y):
            raise ValueError(f"Kích thước X và y không khớp: {len(X)} vs {len(y)}")
        
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        
        if self.n_classes < 3:
            warnings.warn("OVA được thiết kế cho dữ liệu có từ 3 lớp trở lên.")
            
        self.classifiers = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_ova_binary_model)(self.base_estimator, X, y, class_label, idx, self.n_classes, self.verbose)
            for idx, class_label in enumerate(self.classes_)
        )
            
        return self
