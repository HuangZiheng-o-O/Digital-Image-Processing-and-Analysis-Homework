import cv2
import numpy as np
import os

# 定义目录路径（请根据您的实际路径进行修改）
dir_path = 'LabPicsV1/Complex/Test/Koen2All_Chemistry experiment 2. - Coloured flask.-screenshot'

# 验证图像文件是否存在
image_path = os.path.join(dir_path, 'Image.png')
if not os.path.exists(image_path):
    print(f"图像文件不存在：{image_path}")
    exit(1)
image = cv2.imread(image_path)
if image is None:
    print(f"无法读取图像文件：{image_path}")
    exit(1)

height, width, _ = image.shape
final_mask = np.zeros((height, width), dtype=np.uint8)

# 要处理的文件夹列表
instance_folders = ['Material', 'Vessel', 'Parts', 'EmptyRegions']

for folder_name in instance_folders:
    folder_path = os.path.join(dir_path, folder_name)
    if not os.path.exists(folder_path):
        print(f"文件夹不存在：{folder_path}")
        continue
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            mask_path = os.path.join(folder_path, filename)
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                print(f"无法读取掩码文件：{mask_path}")
                continue
            # 使用蓝色通道（第一个通道）作为掩码
            mask_channel = mask[:, :, 0]
            # 根据数值大于 0 的像素确定前景
            instance_mask = (mask_channel > 0).astype(np.uint8)
            # 合并到最终掩码
            final_mask = cv2.bitwise_or(final_mask, instance_mask)

# 应用忽略区域掩码
ignore_path = os.path.join(dir_path, 'Ignore.png')
if os.path.exists(ignore_path):
    ignore_mask = cv2.imread(ignore_path, cv2.IMREAD_UNCHANGED)
    if ignore_mask is not None:
        # 检查掩码的维度
        if len(ignore_mask.shape) == 3:
            # 三通道图像，使用第一个通道
            ignore_mask_channel = ignore_mask[:, :, 0]
        elif len(ignore_mask.shape) == 2:
            # 二维图像，直接使用
            ignore_mask_channel = ignore_mask
        else:
            print(f"Unexpected ignore_mask dimensions: {ignore_mask.shape}")
            exit(1)
        # 将掩码转换为二值图像
        ignore_mask_binary = (ignore_mask_channel > 0).astype(np.uint8)
        # 反转忽略掩码，使忽略区域为0
        ignore_mask_inv = cv2.bitwise_not(ignore_mask_binary)
        # 将忽略掩码应用到最终掩码
        final_mask = cv2.bitwise_and(final_mask, ignore_mask_inv)
    else:
        print(f"无法读取忽略掩码文件：{ignore_path}")

# 将最终掩码转换为三通道 RGB 图像
final_mask_rgb = cv2.cvtColor(final_mask * 255, cv2.COLOR_GRAY2RGB)

# 设置前景像素为白色，背景像素为黑色
foreground = (final_mask > 0)
final_mask_rgb[foreground] = [255, 255, 255]  # 白色
final_mask_rgb[~foreground] = [0, 0, 0]       # 黑色

# 保存最终的掩码图像
output_path = os.path.join(dir_path, 'FinalMask2.png')
cv2.imwrite(output_path, final_mask_rgb)
print(f"最终掩码已保存到 {output_path}")
