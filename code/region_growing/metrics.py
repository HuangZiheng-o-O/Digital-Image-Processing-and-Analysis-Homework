# metrics.py

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

def calculate_metrics(true_mask, predicted_mask):
    """
    计算分割结果的性能指标。

    参数：
        true_mask: 真实的分割掩膜（二值图像，numpy数组）。
        predicted_mask: 预测的分割掩膜（二值图像，numpy数组）。

    返回：
        metrics_text: 包含准确率、精确率和召回率的字符串。
    """
    # 将掩膜展开为一维数组
    true_flat = true_mask.flatten()
    predicted_flat = predicted_mask.flatten()

    # 计算指标
    accuracy = accuracy_score(true_flat, predicted_flat)
    precision = precision_score(true_flat, predicted_flat, zero_division=0)
    recall = recall_score(true_flat, predicted_flat, zero_division=0)

    # 格式化输出
    metrics_text = f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}"
    return metrics_text
