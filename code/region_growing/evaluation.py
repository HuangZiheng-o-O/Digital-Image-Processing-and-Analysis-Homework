import cv2
import numpy as np
import matplotlib.pyplot as plt


# 自动选取种子点
def select_seed_points(image, num_seeds=6):
    """
    自动选取多个种子点，使用局部平均值判定法。

    参数：
        image: 输入的灰度图像（numpy数组）。
        num_seeds: 种子点数量。

    返回：
        seed_points: 种子点的坐标列表。
    """
    height, width = image.shape
    J = np.zeros((height, width))  # 记录已经选取的种子点
    seed_points = []

    for _ in range(num_seeds):
        max_mean = -1
        seed_x, seed_y = -1, -1

        for i in range(1, height - 1, 3):  # 3x3的滑动窗口遍历
            for j in range(1, width - 1, 3):
                local_region = image[i - 1:i + 2, j - 1:j + 2]
                local_mean = np.mean(local_region)

                if local_mean > max_mean and J[i, j] == 0:
                    max_mean = local_mean
                    seed_x, seed_y = i, j

        # 保存选出的种子点
        seed_points.append((seed_y, seed_x))
        J[seed_x, seed_y] = 1  # 标记已经使用的种子点

    return seed_points


# 区域生长算法实现
def region_growing(image, seed_points, threshold=13):
    """
    区域生长算法，使用多个自动选取的种子点。

    参数：
        image: 输入的灰度图像。
        seed_points: 自动选取的种子点列表。
        threshold: 灰度值差异的阈值。

    返回：
        seg_image: 分割后的二值图像。
    """
    height, width = image.shape
    seg_image = np.zeros((height, width), np.uint8)
    visited = np.zeros((height, width), np.bool_)

    seed_list = seed_points[:]
    region_mean = np.mean([float(image[y, x]) for x, y in seed_points])

    for x, y in seed_points:
        visited[y, x] = True
        seg_image[y, x] = 1

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while seed_list:
        x, y = seed_list.pop(0)
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and not visited[ny, nx]:
                pixel_value = float(image[ny, nx])
                if abs(pixel_value - region_mean) <= threshold:
                    seg_image[ny, nx] = 1
                    seed_list.append((nx, ny))
                    region_mean = (region_mean * np.sum(seg_image) + pixel_value) / (np.sum(seg_image) + 1)
                visited[ny, nx] = True

    return seg_image


# SNR计算
def calculate_snr(image, segmented_image):
    """
    计算信噪比（SNR）。
    """
    foreground = image[segmented_image == 1]
    background = image[segmented_image == 0]

    mean_foreground = np.mean(foreground)
    mean_background = np.mean(background)

    std_foreground = np.std(foreground)
    std_background = np.std(background)

    snr = (mean_foreground - mean_background) / (std_foreground + std_background)
    return snr


# 显示分割结果
def show_segmentation(original_image, segmented_image):
    """
    显示原始图像和分割图像。
    """
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title("Segmented Image")
    plt.imshow(segmented_image, cmap='gray')

    plt.show()


import cv2
import os


def main():
    # 检查文件路径是否正确
    img_path = 'images/input_image/org.jpg'

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found at {img_path}")

    # 加载灰度图像
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError("Failed to load the image. Please check the file path and file format.")

    # 自动选取多个种子点
    seed_points = select_seed_points(image, num_seeds=200)

    # 区域生长分割
    segmented_image = region_growing(image, seed_points, threshold=13)

    # 显示结果
    show_segmentation(image, segmented_image)

    # 计算并输出SNR
    snr = calculate_snr(image, segmented_image)
    print(f"信噪比 (SNR): {snr}")


if __name__ == "__main__":
    main()
