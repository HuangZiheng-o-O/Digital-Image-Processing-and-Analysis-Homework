# main.py

import cv2
import numpy as np
from seed_selection import select_seed_point
from region_growing import region_growing
from visualization import display_results_origin
from metrics import calculate_metrics

def main():
    # 输入图像路径
    image_path = 'images/input_image/fruit.png'

    # 读取原始图像和真实掩膜（用于计算指标）
    original_image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # 如果有真实掩膜，读取真实掩膜
    true_mask_path = 'images/true_mask.png'  # 如果没有真实掩膜，可以忽略
    true_mask = cv2.imread(true_mask_path, cv2.IMREAD_GRAYSCALE) if true_mask_path else None
    if true_mask is not None:
        # 将掩膜二值化
        _, true_mask = cv2.threshold(true_mask, 127, 1, cv2.THRESH_BINARY)

    # 交互式选择种子点
    seed_point = select_seed_point(image_path)
    if not seed_point:
        print("未选择种子点，程序退出。")
        return

    # 执行区域生长算法
    segmented_image = region_growing(gray_image, seed_point, threshold=10)

    # 显示结果
    display_results_origin(original_image, segmented_image)

    # 如果有真实掩膜，计算性能指标
    if true_mask is not None:
        metrics_text = calculate_metrics(true_mask, segmented_image)
        print("性能指标：")
        print(metrics_text)
    else:
        print("未提供真实掩膜，无法计算性能指标。")

if __name__ == "__main__":
    main()
