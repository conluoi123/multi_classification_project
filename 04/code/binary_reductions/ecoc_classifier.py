import numpy as np
from typing import Optional, Callable, Literal
from copy import deepcopy
import warnings


class DummyClassifier:
    """Bộ phân loại giả định dùng để dự phòng khi một cột mã hóa chỉ chứa 1 nhãn lớp duy nhất."""
    def __init__(self, prediction=-1):
        self.prediction = prediction
    def fit(self, X, y): return self
    def predict(self, X): return np.full(len(X), self.prediction)
    def decision_function(self, X): return np.full(len(X), self.prediction * 1.0)


class ECOCClassifier:
    """
    Bộ phân loại đa lớp sử dụng mã đầu ra sửa lỗi (Error-Correcting Output Codes).
    """
    def __init__(
        self,
        base_estimator,
        code_matrix: Optional[np.ndarray] = None,
        code_type: Literal['binary', 'ternary'] = 'binary',
        code_size: Optional[int] = None,
        distance_metric: str = 'hamming',
        use_confidence: bool = True,
        verbose: bool = False
    ):
        self.base_estimator = base_estimator
        self.code_matrix = code_matrix
        self.code_type = code_type
        self.code_size = code_size
        self.distance_metric = distance_metric
        self.use_confidence = use_confidence
        self.verbose = verbose
        
        self.classifiers = []
        self.n_classes = None
        self.classes_ = None
        self.fitted_code_matrix_ = None
        
    def _generate_code_matrix(
        self,
        n_classes: int,
        code_size: Optional[int] = None,
        code_type: str = 'binary'
    ) -> np.ndarray:
        """Sinh ma trận mã hóa thông minh, nỗ lực tối đa hóa khoảng cách Hamming."""
        if code_size is None:
            # Chiều dài mã mặc định dựa trên số lượng lớp
            code_size = int(np.ceil(np.log2(n_classes))) + n_classes
            
        best_M = None
        max_min_dist = -1
        
        # Thử nghiệm 100 lần để tìm ma trận có khoảng cách Hamming tối thiểu lớn nhất
        for attempt in range(100):
            if code_type == 'binary':
                M = np.random.choice([-1, 1], size=(n_classes, code_size))
            elif code_type == 'ternary':
                M = np.random.choice([-1, 0, 1], size=(n_classes, code_size))
            else:
                raise ValueError(f"Kiểu mã hóa không hợp lệ: {code_type}")
                
            d = self._minimum_hamming_distance(M)
            if d > max_min_dist:
                max_min_dist = d
                best_M = M
                # Dừng sớm nếu khoảng cách đạt yêu cầu lý tưởng
                if d >= code_size // 2: 
                    break
                    
        return best_M
    
    def _minimum_hamming_distance(self, M: np.ndarray) -> int:
        """Tính khoảng cách Hamming tối thiểu giữa 2 từ mã bất kỳ trong ma trận."""
        k = M.shape[0]
        min_dist = np.inf
        
        for i in range(k):
            for j in range(i+1, k):
                # Chỉ tính toán trên các vị trí mà cả 2 lớp đều tham gia (không bằng 0)
                mask = (M[i] != 0) & (M[j] != 0)
                dist = np.sum(M[i, mask] != M[j, mask])
                min_dist = min(min_dist, dist)
                
        return int(min_dist)
    
    def get_min_hamming_distance(self):
        """Trả về khoảng cách Hamming tối thiểu d và khả năng sửa lỗi r0."""
        if self.fitted_code_matrix_ is None:
            return 0, 0
        d = self._minimum_hamming_distance(self.fitted_code_matrix_)
        r0 = (d - 1) // 2
        return d, int(r0)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Huấn luyện bộ phân loại ECOC."""
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        
        # Sinh ma trận mã hóa nếu chưa có
        if self.code_matrix is None:
            self.fitted_code_matrix_ = self._generate_code_matrix(
                self.n_classes, self.code_size, self.code_type
            )
        else:
            self.fitted_code_matrix_ = self.code_matrix
            
        k, c = self.fitted_code_matrix_.shape
        
        if k != self.n_classes:
            raise ValueError(f"Ma trận mã có {k} hàng nhưng tập dữ liệu có {self.n_classes} lớp")
        
        if self.verbose:
            d, r0 = self.get_min_hamming_distance()
            print(f"Đang huấn luyện ECOC với hình dạng ma trận mã: ({k}, {c})")
            print(f"Khoảng cách Hamming tối thiểu: {d}")
            print(f"Khả năng sửa lỗi: {r0} lỗi")
        
        self.classifiers = []
        
        # Huấn luyện c bộ phân loại nhị phân tương ứng với c cột của ma trận
        for col_idx in range(c):
            if self.verbose:
                print(f"  Đang huấn luyện bộ phân loại thứ {col_idx+1}/{c}...")
            
            column = self.fitted_code_matrix_[:, col_idx]
            y_binary = self._get_column_labels(y, column)
            
            # Lọc bỏ các mẫu có nhãn 0 (dành cho mã Ternary)
            mask = (y_binary != 0)
            X_filtered = X[mask]
            y_filtered = y_binary[mask]
            
            # Xử lý trường hợp cột chỉ chứa 1 nhãn lớp
            if len(np.unique(y_filtered)) < 2:
                pred = y_filtered[0] if len(y_filtered) > 0 else 1
                self.classifiers.append(DummyClassifier(prediction=pred))
                continue
            
            classifier = deepcopy(self.base_estimator)
            try:
                classifier.fit(X_filtered, y_filtered)
                self.classifiers.append(classifier)
            except Exception as e:
                warnings.warn(f"Lỗi khi huấn luyện bộ phân loại thứ {col_idx}: {e}")
                self.classifiers.append(DummyClassifier(prediction=1))
        
        if self.verbose:
            print("Quá trình huấn luyện ECOC hoàn tất.")
        
        return self
    
    def _get_column_labels(self, y: np.ndarray, column: np.ndarray) -> np.ndarray:
        """Chuyển đổi nhãn đa lớp sang nhãn nhị phân {-1, 0, 1} dựa trên cột mã hóa."""
        y_binary = np.zeros(len(y))
        for class_idx, class_label in enumerate(self.classes_):
            mask = (y == class_label)
            y_binary[mask] = column[class_idx]
        return y_binary
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán nhãn lớp bằng cách giải mã (decoding)."""
        code_words = self._predict_code_words(X)
        y_pred_indices = self._decode(code_words)
        return self.classes_[y_pred_indices]
    
    def _predict_code_words(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán các từ mã h(x) (có thể là giá trị Margin hoặc nhãn nhị phân)."""
        n_test = len(X)
        c = len(self.classifiers)
        code_words = np.zeros((n_test, c))
        
        for col_idx, classifier in enumerate(self.classifiers):
            # Ưu tiên sử dụng decision_function để lấy độ tin cậy (Soft Decoding)
            if self.use_confidence and hasattr(classifier, 'decision_function'):
                code_words[:, col_idx] = classifier.decision_function(X)
            else:
                code_words[:, col_idx] = classifier.predict(X)
                
        return code_words
    
    def _decode(self, code_words: np.ndarray) -> np.ndarray:
        """Giải mã từ mã dự đoán để tìm ra chỉ số lớp có khoảng cách gần nhất."""
        M = self.fitted_code_matrix_
        
        if self.use_confidence:
            distances = self._exponential_loss_distance(code_words, M)
        elif self.distance_metric == 'hamming':
            distances = self._hamming_distance(code_words, M)
        elif self.distance_metric == 'euclidean':
            distances = self._euclidean_distance(code_words, M)
        elif callable(self.distance_metric):
            distances = self.distance_metric(code_words, M)
        else:
            raise ValueError(f"Không tìm thấy phương pháp đo khoảng cách: {self.distance_metric}")
        
        return np.argmin(distances, axis=1)

    def _exponential_loss_distance(self, code_words: np.ndarray, M: np.ndarray) -> np.ndarray:
        """
        Hàm giải mã Soft Decoding dựa trên Exponential Loss (Công thức 8.19 trong FML).
        L = sum(exp(-m_lj * f_j(x)))
        """
        n_test = code_words.shape[0]
        k = M.shape[0]
        losses = np.zeros((n_test, k))
        
        for class_idx in range(k):
            mask = (M[class_idx] != 0)
            if np.any(mask):
                margins = M[class_idx, mask] * code_words[:, mask]
                losses[:, class_idx] = np.sum(np.exp(-margins), axis=1)
            else:
                losses[:, class_idx] = np.inf
        return losses
    
    def _hamming_distance(self, code_words: np.ndarray, M: np.ndarray) -> np.ndarray:
        """Tính khoảng cách Hamming (Giải mã cứng - Hard Decoding)."""
        n_test = code_words.shape[0]
        k = M.shape[0]
        distances = np.zeros((n_test, k))
        
        # Chuyển dự đoán về dạng nhị phân {-1, 1}
        code_words_hard = np.sign(code_words)
        code_words_hard[code_words_hard == 0] = 1
        
        for class_idx in range(k):
            mask = (M[class_idx] != 0)
            if np.any(mask):
                distances[:, class_idx] = np.sum(code_words_hard[:, mask] != M[class_idx, mask], axis=1)
            else:
                distances[:, class_idx] = np.inf
                
        return distances
    
    def _euclidean_distance(self, code_words: np.ndarray, M: np.ndarray) -> np.ndarray:
        """Tính khoảng cách Euclidean."""
        n_test = code_words.shape[0]
        k = M.shape[0]
        distances = np.zeros((n_test, k))
        
        for class_idx in range(k):
            mask = (M[class_idx] != 0)
            if np.any(mask):
                distances[:, class_idx] = np.linalg.norm(code_words[:, mask] - M[class_idx, mask], axis=1)
            else:
                distances[:, class_idx] = np.inf
        return distances
    
    def get_code_matrix(self) -> np.ndarray:
        """Trả về ma trận mã hóa đã được huấn luyện."""
        return self.fitted_code_matrix_
    
    def visualize_code_matrix(self):
        """Trực quan hóa ma trận mã hóa dưới dạng bảng ký hiệu."""
        M = self.fitted_code_matrix_
        k, c = M.shape
        d, r0 = self.get_min_hamming_distance()
        
        print("\nTrực quan hóa ma trận mã (Code Matrix):")
        print("="*60)
        print(f"{'Lớp':<8} | Từ mã (Code Word)")
        print("-" * 60)
        for i, class_label in enumerate(self.classes_):
            code_str = ''.join(['+ ' if x > 0 else ('- ' if x < 0 else '0 ') for x in M[i]])
            print(f"{class_label:<8} | {code_str}")
        print("="*60)
        print(f"Chiều dài mã (c): {c}")
        print(f"Khoảng cách Hamming tối thiểu (d): {d}")
        print(f"Khả năng sửa lỗi: lên đến {r0} lỗi")


def create_dense_random_code(n_classes: int, code_length: int) -> np.ndarray:
    """Tạo ma trận mã nhị phân ngẫu nhiên dạng dày (dense)."""
    return np.random.choice([-1, 1], size=(n_classes, code_length))


def create_sparse_random_code(n_classes: int, code_length: int, sparsity: float = 0.3) -> np.ndarray:
    """Tạo ma trận mã tam phân ngẫu nhiên dạng thưa (sparse) với độ thưa xác định."""
    M = np.random.choice(
        [-1, 0, 1],
        size=(n_classes, code_length),
        p=[(1-sparsity)/2, sparsity, (1-sparsity)/2]
    )
    return M