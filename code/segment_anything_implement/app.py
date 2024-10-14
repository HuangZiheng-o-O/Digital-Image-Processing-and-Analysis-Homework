from flask import Flask, render_template, request, jsonify
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Flask 应用程序配置
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 加载 SAM 模型
sam_checkpoint = 'model/model.pth'
model_type = "vit_l"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # 保存上传的图像
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    return jsonify({'image_path': file_path})

@app.route('/process', methods=['POST'])
def process_image():
    data = request.get_json()
    image_path = data['image_path']  # 从客户端请求获取图像路径

    if not os.path.exists(image_path):
        return jsonify({'error': 'Image file not found.'})

    # 读取并处理图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 确保图像在 RGB 模式下处理
    original_shape = image.shape[:2]  # 保存原始图像大小

    # 生成掩码
    masks = mask_generator.generate(image)

    # 创建空白掩码图像
    mask_image = np.zeros_like(image)

    # 可视化掩码
    for mask in masks:
        segmentation = mask['segmentation']
        color_mask = np.random.random((1, 3)).tolist()[0]  # 随机颜色
        mask_image[segmentation] = [int(c * 255) for c in color_mask]  # 应用随机颜色到掩码区域

    # 保存处理后的图像
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], "result.png")
    result_image = Image.fromarray(mask_image)
    result_image = result_image.resize((original_shape[1], original_shape[0]))  # 确保结果与原图尺寸一致
    result_image.save(result_path)

    return jsonify({'result_image_path': result_path})

if __name__ == "__main__":
    app.run(port=5002, debug=True)
