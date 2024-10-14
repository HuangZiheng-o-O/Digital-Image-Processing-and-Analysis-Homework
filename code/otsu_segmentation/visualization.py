import matplotlib.pyplot as plt

def show_segmentation(original_image, segmented_image):
    plt.figure(figsize=(10, 5))

    # 显示原始图像
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(original_image, cmap='gray')
    plt.axis('off')  # 不显示坐标轴

    # 显示分割图像
    plt.subplot(1, 2, 2)
    plt.title('Segmented Image')
    plt.imshow(segmented_image, cmap='gray')
    plt.axis('off')  # 不显示坐标轴

    plt.tight_layout()
    plt.show()




