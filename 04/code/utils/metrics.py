import numpy as np

def accuracy_score(y_true, y_pred):
    """Tính tỷ lệ dự đoán chính xác (Accuracy)."""
    return np.mean(y_true == y_pred)

def pairwise_confusion_matrix(y_true, y_pred, classes=None):
    """
    Tính toán ma trận tỷ lệ dự đoán giữa các cặp lớp để phân tích sai số.
    Hàng i, Cột j thể hiện tỷ lệ mẫu thuộc lớp i bị dự đoán thành lớp j.
    """
    if classes is None:
        classes = np.unique(y_true)
        
    n_classes = len(classes)
    confusion = np.zeros((n_classes, n_classes))
    
    for i in range(n_classes):
        class_i = classes[i]
        mask_i = (y_true == class_i)
        
        if mask_i.sum() == 0:
            continue
        
        for j in range(n_classes):
            class_j = classes[j]
            confusion[i, j] = np.mean(y_pred[mask_i] == class_j)
            
    return confusion
