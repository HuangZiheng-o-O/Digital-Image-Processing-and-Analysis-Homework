import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from PIL import Image
import torch
import torchvision
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# 模型路径和图像路径
image_path = "images/input_image/2092.jpg"
# image_path = "LabPicsV1/Simple/Test/Image/Koen2All_Chemistry experiment 2. - Coloured flask.-screenshot (1).jpg"
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

print(masks)


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
        ax.imshow(np.dstack((img, m**0.5)), alpha=0.6)

# 绘制原图和掩码
fig, axs = plt.subplots(1, 2, figsize=(16, 16))
axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axs[1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
show_anns(masks, axs[1])
axs[0].axis('off')
axs[1].axis('off')
plt.show()
