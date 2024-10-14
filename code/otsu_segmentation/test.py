from segmentation import otsu_thresholding
from visualization import show_segmentation
from utils import load_image
from metrics import calculate_metrics


def main():
    # 加载图像
    image = load_image('/Users/huangziheng/PycharmProjects/otsu_segmentation/img/org.jpg')  # 替换为你的图像路径

    # 应用大津阈值化分割
    segmented_image = otsu_thresholding(image)

    # 显示原始图像和分割结果
    show_segmentation(image, segmented_image)

    # 计算并输出定量分析结果
    accuracy, precision, recall = calculate_metrics(image, segmented_image)
    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')


if __name__ == "__main__":
    main()