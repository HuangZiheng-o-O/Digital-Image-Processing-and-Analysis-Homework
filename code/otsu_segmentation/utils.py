import cv2

def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
