import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 设置环境变量
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 模型路径和图像路径
# image_path = "images/input_image/2092.jpg"
image_path = "LabPicsV1/Simple/Test/Image/Koen2All_Chemistry experiment 2. - Coloured flask.-screenshot (1).jpg"
sam_checkpoint = 'model/model.pth'

# 模型类型与设备
model_type = "vit_l"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载预训练模型
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# 初始化掩码生成器
mask_generator = SamAutomaticMaskGenerator(sam)

# 加载图像并调整大小
image = cv2.imread(image_path)
image = cv2.resize(image, (224, 224))

# 生成掩码
masks = mask_generator.generate(image)


# 可视化结果
def show_anns(anns, ax=None):
    if len(anns) == 0:
        return
    if ax is None:
        ax = plt.gca()
    ax.set_autoscale_on(False)
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack((img, m ** 0.5)), alpha=0.6)


# 绘制原图和掩码
fig, axs = plt.subplots(1, 2, figsize=(16, 16))
axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axs[1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
show_anns(masks, axs[1])
axs[0].axis('off')
axs[1].axis('off')
plt.show()


def read_seg_file(seg_file_path):
    with open(seg_file_path, 'r') as f:
        lines = f.readlines()

    # 解析头部
    header = {}
    for line in lines:
        if line.startswith('data'):
            break
        # 跳过不需要的行
        if line.startswith(('format', 'date', 'image', 'user', 'gray', 'invert', 'flipflop')):
            continue
        try:
            key, value = line.split()
            header[key] = value
        except ValueError:
            print(f"Skipping line in header: {line.strip()}")  # 打印跳过的行

    # 获取图像宽高
    width = int(header['width'])
    height = int(header['height'])

    # 创建空白掩码图像
    ground_truth_mask = np.zeros((height, width), dtype=np.uint8)

    # 解析数据部分
    for line in lines[len(header) + 1:]:
        if line.strip():  # 确保不是空行
            try:
                s, r, c1, c2 = map(int, line.split())
                ground_truth_mask[r, c1:c2 + 1] = s + 1  # 将像素值设为segment编号
            except ValueError:
                print(f"Skipping line in data: {line.strip()}")  # 打印跳过的行

    return ground_truth_mask


# 读取ground truth掩码
ground_truth_path = "images/true_mask/2092.seg"
ground_truth_mask = read_seg_file(ground_truth_path)

# 与生成的masks进行比较
predicted_mask = np.zeros(ground_truth_mask.shape, dtype=np.uint8)

# 将模型生成的masks转换为预测掩码
for idx, mask in enumerate(masks):
    predicted_mask[mask['segmentation']] = idx + 1

# 计算定量指标
accuracy = accuracy_score(ground_truth_mask.flatten(), predicted_mask.flatten())
precision = precision_score(ground_truth_mask.flatten(), predicted_mask.flatten(), average='weighted', zero_division=0)
recall = recall_score(ground_truth_mask.flatten(), predicted_mask.flatten(), average='weighted', zero_division=0)
f1 = f1_score(ground_truth_mask.flatten(), predicted_mask.flatten(), average='weighted', zero_division=0)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# 可视化结果
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axs[0].set_title('Original Image')
axs[1].imshow(ground_truth_mask, cmap='jet', alpha=0.5)
axs[1].set_title('Ground Truth Mask')
axs[2].imshow(predicted_mask, cmap='jet', alpha=0.5)
axs[2].set_title('Predicted Mask')
for ax in axs:
    ax.axis('off')
plt.show()
