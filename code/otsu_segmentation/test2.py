import cv2
import numpy as np

from utils import load_image


def otsu_thresholding2(image):
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

    return thresh_image, threshold

def calculate_snr(image, segmented_image):

    foreground = image[segmented_image == 255]

    background = image[segmented_image == 0]

    mean_foreground = np.mean(foreground)
    mean_background = np.mean(background)
    std_foreground = np.std(foreground)
    std_background = np.std(background)

    # 计算SNR
    snr = (mean_foreground - mean_background) / (std_foreground + std_background)

    return snr

def main():
    # 加载图像
    image = load_image('/Users/huangziheng/PycharmProjects/otsu_segmentation/img/org.jpg')  # 替换为你的图像路径

    # 应用大津阈值化分割
    segmented_image, threshold = otsu_thresholding2(image)


    # 计算并输出定量分析结果（SNR）
    snr = calculate_snr(image, segmented_image)
    print(f"Otsu 阈值: {threshold}")
    print(f"信噪比 (SNR): {snr}")

if __name__ == "__main__":
    main()
