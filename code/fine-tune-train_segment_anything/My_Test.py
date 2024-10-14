import numpy as np
import torch
import cv2
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt


# 定义掩码匹配函数（使用Hungarian算法）
def match_masks(pred_masks, gt_masks):
    """
    使用Hungarian算法匹配预测掩码和真实掩码，基于IoU最大化匹配。

    pred_masks: Tensor of shape (num_preds, H, W)
    gt_masks: Tensor of shape (num_gts, H, W)

    返回:
        row_ind: List of ground truth indices
        col_ind: List of predicted mask indices
        matched_ious: List of IoU values for matched pairs
    """
    num_preds = pred_masks.size(0)
    num_gts = gt_masks.size(0)

    if num_preds == 0 or num_gts == 0:
        return [], [], []

    iou_matrix = torch.zeros((num_gts, num_preds), device=pred_masks.device)

    for gt_idx in range(num_gts):
        for pred_idx in range(num_preds):
            intersection = (gt_masks[gt_idx] * (pred_masks[pred_idx] > 0.5)).sum()
            union = gt_masks[gt_idx].sum() + (pred_masks[pred_idx] > 0.5).sum() - intersection
            iou = intersection / union if union > 0 else 0.0
            iou_matrix[gt_idx, pred_idx] = iou

    # 转换为numpy
    iou_matrix_np = iou_matrix.cpu().detach().numpy()

    # 使用Hungarian算法找到最佳匹配
    row_ind, col_ind = linear_sum_assignment(-iou_matrix_np)  # 最大化IoU，因此取负值

    # 获取匹配后的IoU值
    matched_ious = iou_matrix_np[row_ind, col_ind]

    return row_ind.tolist(), col_ind.tolist(), matched_ious.tolist()


# 定义指标计算函数
def calculate_iou(pred_mask, gt_mask):
    """
    计算单个掩码的IoU。

    pred_mask: numpy数组，二值化的预测掩码
    gt_mask: numpy数组，二值化的真实掩码

    返回:
        IoU值
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = intersection / union if union > 0 else 0.0
    return iou


# 定义可视化函数
def visualize_masks(image, gt_masks, pred_masks, save_path="visualization.png"):
    """
    可视化真实掩码和预测掩码。

    image: 原始图像，numpy数组
    gt_masks: 真实掩码，numpy数组，形状为(num_gts, H, W)
    pred_masks: 预测掩码，numpy数组，形状为(num_preds, H, W)
    save_path: 保存路径
    """
    # 生成随机颜色
    num_gts = gt_masks.shape[0]
    num_preds = pred_masks.shape[0]
    colors_gt = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(num_gts)]
    colors_pred = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(num_preds)]

    # 绘制真实掩码
    image_gt = image.copy()
    for idx, mask in enumerate(gt_masks):
        color = colors_gt[idx]
        image_gt[mask.astype(bool)] = image_gt[mask.astype(bool)] * 0.5 + np.array(color) * 0.5

    # 绘制预测掩码
    image_pred = image.copy()
    for idx, mask in enumerate(pred_masks):
        color = colors_pred[idx]
        image_pred[mask.astype(bool)] = image_pred[mask.astype(bool)] * 0.5 + np.array(color) * 0.5

    # 合并可视化
    combined_image = np.hstack((image_gt, image_pred))

    # 显示并保存
    plt.figure(figsize=(20, 10))
    plt.imshow(combined_image.astype(np.uint8))
    plt.axis('off')
    plt.title('左边: 真实掩码 | 右边: 预测掩码')
    plt.savefig(save_path)
    plt.show()


# 定义测试函数
def test_single_image(model, image_path, mask_path, device="cuda"):
    """
    测试单张图像，计算指标并可视化结果。

    model: 训练好的模型
    image_path: 测试图像路径
    mask_path: 真实掩码路径
    device: 设备
    """
    # 读取图像和掩码
    img = cv2.imread(image_path)[..., ::-1]  # BGR to RGB
    mask = cv2.imread(mask_path, 0)  # 读取为灰度图

    # 预处理
    r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])  # 缩放因子
    img_resized = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    mask_resized = cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)), interpolation=cv2.INTER_NEAREST)

    # 合并掩码（与训练一致）
    mat_map = mask_resized.copy()
    mat_map[mat_map == 0] = 0  # 假设只有材料和血管
    inds = np.unique(mat_map)[1:]  # 排除背景
    points = []
    masks = []
    for ind in inds:
        binary_mask = (mat_map == ind).astype(np.uint8)
        masks.append(binary_mask)
        coords = np.argwhere(binary_mask > 0)
        if len(coords) == 0:
            continue
        yx = coords[np.random.randint(len(coords))]
        points.append([[yx[1], yx[0]]])  # (x, y)

    if len(masks) == 0:
        print("掩码中未找到任何物体。")
        return

    # 转换为numpy数组
    masks = np.array(masks)
    points = np.array(points)
    input_labels = np.ones((len(masks), 1))  # 全为正样本

    # 生成提示点
    input_points = points

    # 将掩码转换为Tensor
    gt_mask_tensor = torch.tensor(masks.astype(np.float32)).to(device)  # (num_gts, H, W)

    # 运行模型预测
    with torch.no_grad():
        model.set_image(img_resized)
        masks_pred, scores_pred, logits_pred = model.predict(
            point_coords=input_points,
            point_labels=input_labels
        )

    # 提取预测掩码和分数
    masks_pred = masks_pred[:, 0].astype(bool)  # 选择第一个掩码
    scores_pred = scores_pred[:, 0].numpy()

    # 转换为Tensor
    pred_mask_tensor = torch.tensor(masks_pred.astype(np.float32)).to(device)  # (num_preds, H, W)

    # 匹配掩码
    row_ind, col_ind, matched_ious = match_masks(pred_mask_tensor, gt_mask_tensor)

    # 计算IoU
    iou_scores = matched_ious  # 已经是匹配后的IoU

    # 计算平均IoU
    average_iou = np.mean(iou_scores) if len(iou_scores) > 0 else 0.0
    print(f"图像: {os.path.basename(image_path)} | 平均IoU: {average_iou:.4f}")

    # 可视化结果
    visualize_masks(img_resized, masks, masks_pred, save_path=f"visualization_{os.path.basename(image_path)}.png")

    # 返回详细指标
    return average_iou, iou_scores


# 定义批量测试函数
def test_multiple_images(model, test_data, device="cuda"):
    """
    测试多张图像，计算并汇总指标。

    model: 训练好的模型
    test_data: List of tuples，格式为[(image_path, mask_path), ...]
    device: 设备
    """
    total_iou = []
    iou_scores_all = []

    for idx, (image_path, mask_path) in enumerate(test_data):
        print(f"正在测试第 {idx + 1}/{len(test_data)} 张图像: {image_path}")
        average_iou, iou_scores = test_single_image(model, image_path, mask_path, device=device)
        total_iou.append(average_iou)
        iou_scores_all.extend(iou_scores)

    if len(total_iou) > 0:
        overall_average_iou = np.mean(total_iou)
    else:
        overall_average_iou = 0.0

    print(f"\n整体测试集平均IoU: {overall_average_iou:.4f}")
    return overall_average_iou, iou_scores_all


# 主函数
if __name__ == "__main__":
    # 配置路径
    data_dir = r"LabPicsV1//"  # 数据集根目录
    model_checkpoint = "model.torch"  # 训练后保存的模型权重路径
    model_cfg = "sam2_hiera_s.yaml"  # 模型配置文件

    # 准备测试数据列表
    # 假设测试图像位于 "LabPicsV1/Simple/Test/Image/"，对应掩码位于 "LabPicsV1/Simple/Test/Instance/"
    test_image_dir = os.path.join(data_dir, "Simple/Test/Image/")
    test_mask_dir = os.path.join(data_dir, "Simple/Test/Instance/")

    image_name  = "Koen2All_Chemistry experiment 2. - Coloured flask.-screenshot (1).jpg"  # 测试图像文件名
    image_path = os.path.join(test_image_dir, image_name)
    mask_name = os.path.splitext(image_name)[0] + ".png"  # 假设掩码为PNG格式
    mask_path = os.path.join(test_mask_dir, mask_name)


    # 加载模型
    sam2_model = build_sam2(model_cfg, model_checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)

    # 加载训练好的权重
    predictor.model.load_state_dict(torch.load(model_checkpoint))
    predictor.model.to("cuda")
    predictor.model.eval()  # 设置为评估模式

    # 运行测试def
    overall_iou, all_iou_scores = test_single_image(predictor, image_path, mask_path, device="cuda")
    # overall_iou, all_iou_scores = test_multiple_images(predictor, test_data, device="cuda")

    print(f"\n测试图像IoU: {overall_iou:.4f}")
