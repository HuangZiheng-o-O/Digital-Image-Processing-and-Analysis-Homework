from region_growing import region_growing
import cv2
import matplotlib.pyplot as plt
import numpy as np


def select_seed_points(image_shape, num_points):
    height, width = image_shape
    y_coords = np.linspace(0, height - 1, int(np.sqrt(num_points)), dtype=int)
    x_coords = np.linspace(0, width - 1, int(np.sqrt(num_points)), dtype=int)
    return [(x, y) for y in y_coords for x in x_coords]


if __name__ == "__main__":
    # 读取灰度图像
    image = cv2.imread('./images/input_image/fruit.png', cv2.IMREAD_GRAYSCALE)

    # 获取图像大小
    image_shape = image.shape

    # 生成均匀分布的种子点
    num_points = 25  # 选择的种子点数量
    seed_points = select_seed_points(image_shape, num_points)

    # 执行区域生长
    seg_images = []
    for seed_point in seed_points:
        seg_image = region_growing(image, seed_point, threshold=10)
        seg_images.append(seg_image)

    # 显示结果
    plt.figure(figsize=(15, 10))
    plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('原始图像')
    plt.subplot(122), plt.imshow(np.max(seg_images, axis=0), cmap='gray'), plt.title('分割结果')
    plt.show()
