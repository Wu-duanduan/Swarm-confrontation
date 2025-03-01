import cv2
import numpy as np
from PIL import Image as PILImage

def get_rendered_image(color_buffer):
    # 获取图像的原始像素数据
    image_data = color_buffer.get_image_data()

    # 将数据转换为 NumPy 数组（需要转换为 RGB 格式）
    img_data = np.frombuffer(image_data.get_data('RGB', image_data.width * 3), dtype=np.uint8)
    img_data = img_data.reshape((image_data.height, image_data.width, 3))

    # 翻转图像的垂直方向
    img_data = np.flipud(img_data)  # 或者 img_data = img_data[::-1]
    return img_data


def locate_cars(image, img_height, img_width, save_flag=False, save_path=None):
    image = np.copy(image)
    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # 定义红色的HSV范围
    # lower_red1 = np.array([0, 50, 50])
    # upper_red1 = np.array([10, 255, 255])
    lower_red = np.array([0, 255, 0])  # 红色低值范围
    upper_red = np.array([10, 255, 255])  # 红色低值范围
    # 较亮的红色
    # lower_red2 = np.array([170, 70, 50])
    # upper_red2 = np.array([180, 255, 255])

    # 定义蓝色的HSV范围
    # lower_blue = np.array([90, 70, 50])
    # upper_blue = np.array([130, 255, 255])
    lower_blue = np.array([100, 255, 255])  # 蓝色低值范围
    upper_blue = np.array([140, 255, 255])  # 蓝色高值范围
    # 创建掩码
    # mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    # mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    # mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_red = cv2.inRange(hsv, lower_red, upper_red)

    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # 找到红车和蓝车的轮廓
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 提取红车和蓝车的位置
    red_car_positions = []
    blue_car_positions = []

    for contour in contours_blue:
        x, y, w, h = cv2.boundingRect(contour)
        # blue_car_positions.append((x, y))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        x_origin, y_origin = convert_coordinates(x + w // 2, y + h // 2, img_height, img_width)
        blue_car_positions.append((x_origin, y_origin))
        cv2.putText(image, f"{x_origin, y_origin}", (x-30, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)

    for contour in contours_red:
        x, y, w, h = cv2.boundingRect(contour)
        # red_car_positions.append((x, y))

        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        x_origin, y_origin = convert_coordinates(x + w // 2, y + h // 2, img_height, img_width)
        red_car_positions.append((x_origin, y_origin))
        # cv2.putText(image, f"({x + w // 2}, {y + h // 2})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #             (0, 0, 255), 2)
        cv2.putText(image, f"{x_origin, y_origin}", (x-30, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 0), 2)

    if save_flag:
        img = PILImage.fromarray(image)
        img.save(save_path)
        # cv2.imwrite(save_path, image)
        # print("Image saved to",save_path)

    # 打印结果
    # print("红车位置:", red_car_positions)
    # print("蓝车位置:", blue_car_positions)
    return red_car_positions, blue_car_positions


def convert_coordinates(x, y, img_height, img_width):
    """
    将像素坐标系的(x, y)转换为(-15, +15, -8, +8)坐标系的坐标
    """
    # 计算转换后的坐标
    # print("img_height:", img_height)
    # print("img_width:", img_width)
    x_scale = 30 / img_width
    y_scale = 16 / img_height

    # 转换坐标
    x_target = x * x_scale - 15
    y_target = -y * y_scale + 8

    # 保留两位小数
    x_target = round(x_target, 2)
    y_target = round(y_target, 2)

    return x_target, y_target
