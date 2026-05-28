import numpy as np


def gini(y):
    m = len(y)
    if m == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / m
    return 1.0 - np.sum(p ** 2)


def entropy(y):
    m = len(y)
    if m == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / m
    return -np.sum(p * np.log2(p + 1e-12))


def best_split(X, y, criterion='gini'):
    impurity_fn = gini if criterion == 'gini' else entropy
    best_gain, best_j, best_t = -np.inf, None, None
    parent_imp = impurity_fn(y)
    m, n_feat = X.shape

    for j in range(n_feat):
        thresholds = np.unique(X[:, j])
        for t in thresholds:
            left_mask = X[:, j] <= t
            right_mask = X[:, j] > t
            left_y = y[left_mask]
            right_y = y[right_mask]

            if len(left_y) == 0 or len(right_y) == 0:
                continue

            gain = parent_imp - (
                (len(left_y) / m) * impurity_fn(left_y) +
                (len(right_y) / m) * impurity_fn(right_y)
            )

            if gain > best_gain:
                best_gain = gain
                best_j = j
                best_t = t

    return best_j, best_t, best_gain


class _Node:
    """Nút trong cây quyết định (nút trong hoặc lá)."""
    def __init__(self, label=None, feature=None, threshold=None,
                 left=None, right=None):
        self.label = label          # Nhãn lá, None nếu là nút trong
        self.feature = feature      # Chỉ số đặc trưng dùng để tách
        self.threshold = threshold  # Ngưỡng tách: X[:, feature] <= threshold → trái
        self.left = left            # Nút con trái  (X[:, j] <= t)
        self.right = right          # Nút con phải (X[:, j] >  t)

    def is_leaf(self):
        return self.label is not None


class DecisionTreeClassifier:
    """Cây quyết định nhị phân cài từ đầu."""

    def __init__(self, criterion='gini', max_depth=None, min_samples=2):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root = None
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.root = self._build(X, y, depth=0)
        return self

    def _majority(self, y):
        """Trả về nhãn chiếm đa số (majority vote)."""
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def _build(self, X, y, depth):
        """Đệ quy xây dựng cây từ tập con (X, y) tại độ sâu depth."""
        # --- Điều kiện dừng ---
        # 1. Tất cả mẫu cùng lớp
        if len(np.unique(y)) == 1:
            return _Node(label=y[0])

        # 2. Đạt độ sâu tối đa
        if self.max_depth is not None and depth >= self.max_depth:
            return _Node(label=self._majority(y))

        # 3. Quá ít mẫu để tách tiếp
        if len(y) < self.min_samples:
            return _Node(label=self._majority(y))

        # --- Tìm tách tốt nhất ---
        j, t, gain = best_split(X, y, criterion=self.criterion)

        # Không tìm được tách nào có ích (gain <= 0 hoặc không split được)
        if j is None or gain <= 0:
            return _Node(label=self._majority(y))

        # --- Chia nút ---
        left_mask = X[:, j] <= t
        right_mask = ~left_mask

        left_node = self._build(X[left_mask], y[left_mask], depth + 1)
        right_node = self._build(X[right_mask], y[right_mask], depth + 1)

        return _Node(feature=j, threshold=t, left=left_node, right=right_node)

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])

    def _predict_one(self, x, node):
        """Đi từ gốc xuống lá để dự đoán nhãn cho một điểm x."""
        if node.is_leaf():
            return node.label
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def impurity_per_depth(self, X, y):
        """
        Trả về danh sách impurity trung bình của các lá theo độ sâu cây.
        Dùng để kiểm chứng tính đơn điệu giảm impurity (mục 1.3.2).
        """
        impurity_fn = gini if self.criterion == 'gini' else entropy
        records = []
        self._collect_leaves(self.root, X, y, depth=0,
                             impurity_fn=impurity_fn, records=records)
        # Nhóm theo độ sâu, lấy trung bình có trọng số theo số mẫu
        from collections import defaultdict
        depth_data = defaultdict(lambda: [0.0, 0])
        for d, imp, n in records:
            depth_data[d][0] += imp * n
            depth_data[d][1] += n
        return {d: v[0] / v[1] for d, v in sorted(depth_data.items())}

    def _collect_leaves(self, node, X, y, depth, impurity_fn, records):
        if node.is_leaf():
            records.append((depth, impurity_fn(y), len(y)))
            return
        mask = X[:, node.feature] <= node.threshold
        self._collect_leaves(node.left,  X[mask],  y[mask],  depth + 1,
                             impurity_fn, records)
        self._collect_leaves(node.right, X[~mask], y[~mask], depth + 1,
                             impurity_fn, records)