import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

def calculate_metrics(original_image, segmented_image):

    y_true = (original_image > 0).astype(int).flatten()  # 将原始图像转为二值
    y_pred = (segmented_image > 0).astype(int).flatten()  # 将分割图像转为二值

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    return accuracy, precision, recall
