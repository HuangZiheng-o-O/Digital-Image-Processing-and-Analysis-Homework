# seed_selection.py

import cv2

def select_seed_point(image_path):
    """
    允许用户在图像上点击以选择种子点。

    参数：
        image_path: 输入图像的路径。

    返回：
        seed_point: 用户选择的种子点坐标，格式为(x, y)。
    """
    # 读取图像
    image = cv2.imread(image_path)
    clone = image.copy()
    seed_point = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 清除之前的种子点
            seed_point.clear()
            seed_point.append((x, y))
            # 在图像上绘制红色小圆点
            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
            cv2.imshow("Select Seed Point", image)

    cv2.namedWindow("Select Seed Point")
    cv2.setMouseCallback("Select Seed Point", mouse_callback)

    print("请在图像上点击以选择种子点，按 'c' 键确认，按 'r' 键重置。")

    while True:
        cv2.imshow("Select Seed Point", image)
        key = cv2.waitKey(1) & 0xFF
        # 按 'r' 键重置图像和种子点
        if key == ord("r"):
            image = clone.copy()
            seed_point.clear()
            print("已重置，请重新选择种子点。")
        # 按 'c' 键确认选择
        elif key == ord("c"):
            if seed_point:
                print(f"已选择种子点：{seed_point[0]}")
                break
            else:
                print("未选择种子点，请点击图像选择。")

    cv2.destroyAllWindows()
    if seed_point:
        return seed_point[0]
    else:
        return None
