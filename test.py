import cv2
import numpy as np
import os


def process_image(image_path, output_path, target_size=(384, 384)):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 去除白色边框，假设白色为(255, 255, 255)
    mask = cv2.inRange(gray, np.array([1]), np.array([145]))
    # cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)

    kernel = np.ones((9, 9), np.uint8)
    # 进行腐蚀操作，去掉外层噪声边缘
    eroded_image = cv2.erode(mask, kernel, iterations=1)
    # 进行膨胀操作，还原内部边缘
    cleaned_image = cv2.dilate(eroded_image, kernel, iterations=1)

    cleaned_image = cv2.dilate(cleaned_image, kernel, iterations=1)

    cleaned_image = cv2.erode(cleaned_image, kernel, iterations=1)

    coords = cv2.findNonZero(cleaned_image)
    cleaned_image = cv2.merge((cleaned_image, cleaned_image, cleaned_image))

    image = cv2.bitwise_and(image, cleaned_image)

    # cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
    # cv2.imshow('mask', cleaned_image)
    # cv2.waitKey(0)

    # 获取非白色像素的边界框
    x, y, w, h = cv2.boundingRect(coords)

    # 裁剪图像
    cropped_image = image[y:y + h, x:x + w]
    # cv2.imshow('c', cropped_image)
    # cv2.waitKey(0)

    # 获取裁剪后图像的大小
    crop_height, crop_width = cropped_image.shape[:2]

    # 如果裁剪后的图像大于目标尺寸，则缩放图像
    if crop_height > target_size[0] or crop_width > target_size[1]:
        scaling_factor = min(target_size[0] / crop_height, target_size[1] / crop_width)
        new_size = (int(crop_width * scaling_factor), int(crop_height * scaling_factor))
        cropped_image = cv2.resize(cropped_image, new_size, interpolation=cv2.INTER_AREA)
        crop_height, crop_width = cropped_image.shape[:2]

    # 创建一个黑色背景的图像
    centered_image = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)

    # 计算放置裁剪图像的起始位置，使其居中
    start_y = (target_size[0] - crop_height) // 2
    start_x = (target_size[1] - crop_width) // 2

    # 将裁剪后的图像放置到黑色背景的中心位置
    centered_image[start_y:start_y + crop_height, start_x:start_x + crop_width] = cropped_image

    # 保存处理后的图像
    cv2.imwrite(output_path, centered_image)


# 示例使用
# input_path = r'D:\IC\HoneyZSY\project\dataset\Leaves\Leaves\Ash\leaflet\leaflet1-2.jpg'
# output_path = r'.\leaflet.jpg'
# process_image(input_path, output_path)

path = r'D:\PycharmProject\Others\BLUE'
dir = 'leaflet2'
cropped_dir = '.'

if not os.path.exists(os.path.join(path, cropped_dir)):
    os.makedirs(os.path.join(path, cropped_dir))

for file in os.listdir(os.path.join(path, dir)):
    if 'leaflet24-4' in file:
        print(file)
        input_path = os.path.join(path, dir, file)
        save_path = os.path.join(path, cropped_dir, file)
        process_image(input_path, save_path)
