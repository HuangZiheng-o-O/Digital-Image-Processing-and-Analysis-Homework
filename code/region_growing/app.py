from flask import Flask, render_template, request, jsonify
import os
import cv2
from region_growing import region_growing
from seed_selection import select_seed_point
from visualization import display_results

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


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
    image_path = data['image_path']  # 此时是相对路径
    print("image_path", image_path)
    x = int(data['x'])  # 确保转换为整数
    y = int(data['y'])  # 确保转换为整数

    seed_point = (x, y)
    print("seed_point", seed_point)

    original_image = cv2.imread(image_path)
    print("1")

    if original_image is None:
        return jsonify({'error': '无法读取图像文件，请检查路径。'})

    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    print("2")
    # 执行区域生长算法
    segmented_image = region_growing(gray_image, seed_point, threshold=10)

    print("wooo")

    # 保存处理后的图像
    result_overlay_path, result_segmented_path = display_results(original_image, segmented_image,
                                                                 app.config['UPLOAD_FOLDER'])
    print("result_overlay_path", result_overlay_path)
    print("result_segmented_path", result_segmented_path)
    return jsonify({
        'result_overlay_path': result_overlay_path,
        'result_segmented_path': result_segmented_path
    })



if __name__ == "__main__":

    app.run(port=5001,debug=True)

