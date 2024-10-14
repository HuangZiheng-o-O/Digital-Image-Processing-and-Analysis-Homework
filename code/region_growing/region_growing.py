# region_growing.py

import numpy as np

def region_growing(image, seed_point, threshold=10):
    """
    对图像进行区域生长分割。

    参数：
        image: 输入的灰度图像（numpy数组）。
        seed_point: 种子点坐标，格式为(x, y)。
        threshold: 灰度值差异的阈值。

    返回：
        seg_image: 分割后的二值图像，区域为1，其余为0。
    """
    height, width = image.shape
    seg_image = np.zeros((height, width), np.uint8)
    visited = np.zeros((height, width), np.bool_)

    # 初始化种子点列表
    seed_list = [seed_point]
    visited[seed_point[1], seed_point[0]] = True
    seg_image[seed_point[1], seed_point[0]] = 1
    region_mean = float(image[seed_point[1], seed_point[0]])

    # 定义四邻域
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
                    # 更新区域均值
                    region_mean = (region_mean * np.sum(seg_image) + pixel_value) / (np.sum(seg_image) + 1)
                visited[ny, nx] = True

    return seg_image
