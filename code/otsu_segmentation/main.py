import gradio as gr
from segmentation import otsu_thresholding
from utils import load_image
from metrics import calculate_metrics
import cv2
import numpy as np
from visualization import show_segmentation

def process_image(image):
    print("Received image for processing")
    # 确保输入图像为灰度图像，若不是则转换
    if len(image.shape) == 3:  # 如果是RGB图像
        print("Converting to grayscale")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print("Applying Otsu's thresholding")
    segmented_image = otsu_thresholding(image)

    print("Showing segmentation")
    show_segmentation(image, segmented_image)

    print("Calculating metrics")
    accuracy, precision, recall = calculate_metrics(image, segmented_image)

    # 格式化结果
    metrics_text = f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}"

    return segmented_image



def main():
    # Gradio 界面
    iface = gr.Interface(
        fn=process_image,
        inputs=gr.Image(label="Upload Image"),
        outputs=[
            gr.Image(label="Segmented Image"),
            # gr.Textbox(label="Metrics")
        ],
        title="Otsu Thresholding Segmentation",
        description="Upload an image to apply Otsu's thresholding method and view the segmented result along with evaluation metrics."
    )
    iface.launch(share=True)


if __name__ == "__main__":
    main()


