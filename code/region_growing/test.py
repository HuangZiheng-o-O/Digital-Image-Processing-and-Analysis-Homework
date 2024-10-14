
from region_growing import region_growing
from seed_selection import select_seed_point
if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt

    # 读取灰度图像
    image = cv2.imread('./images/input_image/fruit.png', cv2.IMREAD_GRAYSCALE)

    # 定义种子点
    seed_point = (1000, 300)

    # 执行区域生长
    seg_image = region_growing(image, seed_point, threshold=10)

    # 显示结果
    plt.figure(figsize=(10, 5))
    plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('原始图像')
    plt.subplot(122), plt.imshow(seg_image, cmap='gray'), plt.title('分割结果')
    plt.show()

# if __name__ == "__main__":
#     image_path = '/images/input_image/fruit.png'
#     seed_point = select_seed_point(image_path)
#     if seed_point:
#         print(f"选定的种子点坐标为：{seed_point}")
#     else:
#         print("未选择种子点。")
