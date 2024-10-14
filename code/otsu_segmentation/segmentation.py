import cv2
import numpy as np


def otsu_thresholding(image):
    # 计算直方图
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    total_pixels = image.size

    current_max, threshold = 0, 0
    sum_total, sum_background = 0, 0
    weight_background, weight_foreground = 0, 0

    for i in range(256):
        sum_total += i * hist[i]

    for i in range(256):
        weight_background += hist[i]
        if weight_background == 0:
            continue

        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0:
            break

        sum_background += i * hist[i]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground

        # 计算类间方差
        between_class_variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

        if between_class_variance > current_max:
            current_max = between_class_variance
            threshold = i

    _, thresh_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)


    # 确保输出为uint8格式
    thresh_image = thresh_image.astype(np.uint8)

    return thresh_image
