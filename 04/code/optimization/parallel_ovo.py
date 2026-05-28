import numpy as np
import warnings
from typing import Any, Optional, List, Tuple, Dict
from copy import deepcopy
from itertools import combinations
from joblib import Parallel, delayed

from src.binary_reductions.ovo_classifier import OneVsOneClassifier

def _fit_ovo_binary_pair(base_estimator, X, y, i, j, class_i, class_j, idx, n_classifiers, verbose):
    if verbose:
        print(f"  [{idx+1}/{n_classifiers}] Đang huấn luyện bộ phân loại cho cặp lớp {class_i} vs {class_j} (Đa luồng)...")

    mask = (y == class_i) | (y == class_j)
    X_pair = X[mask]
    y_pair = y[mask]

    y_binary = np.where(y_pair == class_i, -1, 1)
    
    classifier = deepcopy(base_estimator)
    try:
        classifier.fit(X_pair, y_binary)
        return (i, j), classifier
    except Exception as e:
        warnings.warn(f"Lỗi khi huấn luyện bộ phân loại cho cặp ({i},{j}): {e}")
        return (i, j), None

class ParallelOneVsOneClassifier(OneVsOneClassifier):
    """
    [TỐI ƯU HÓA] Phiên bản kế thừa của OneVsOneClassifier tích hợp xử lý song song (joblib).
    Tuân thủ nguyên tắc OCP: Không sửa đổi mã gốc của thành viên khác.
    """
    def __init__(
        self,
        base_estimator,
        n_jobs: int = -1,
        verbose: bool = False
    ):
        super().__init__(base_estimator, n_jobs, verbose)
        self.n_jobs = n_jobs
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Huấn luyện bộ phân loại OVO song song đa luồng."""
        if len(X) != len(y):
            raise ValueError(f"X và y phải có cùng độ dài: {len(X)} so với {len(y)}")

        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        
        if self.n_classes < 2:
            raise ValueError(f"Cần ít nhất 2 lớp, hiện có {self.n_classes}")

        self.pairwise_indices_ = list(combinations(range(self.n_classes), 2))
        n_classifiers = len(self.pairwise_indices_)
        
        if self.verbose:
            print(f"Đang huấn luyện OVO với {self.n_classes} lớp song song ({self.n_jobs} luồng)...")
            print(f"Số lượng bộ phân loại nhị phân: {n_classifiers}")

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_ovo_binary_pair)(self.base_estimator, X, y, i, j, self.classes_[i], self.classes_[j], idx, n_classifiers, self.verbose)
            for idx, (i, j) in enumerate(self.pairwise_indices_)
        )
        
        self.classifiers = {k: v for k, v in results if v is not None}
        
        if self.verbose:
            print("Quá trình huấn luyện OVO hoàn tất.")
        
        return self
