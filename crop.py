import cv2
import os
import numpy as np


def extract_leaf(image_path, output_size=(384, 384)):
    # 读取图像
    image = cv2.imread(image_path)

    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 设定绿色的阈值范围来提取叶片
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    # 创建掩码来提取叶片
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # 通过掩码提取叶片区域
    result = cv2.bitwise_and(image, image, mask=mask)

    # 找到叶片区域的边界框
    coords = np.column_stack(np.where(mask > 0))
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # 裁剪出叶片区域
    cropped_leaf = result[y_min:y_max, x_min:x_max]

    # 计算缩放比例
    h, w = cropped_leaf.shape[:2]
    aspect_ratio = w / h
    target_width, target_height = output_size

    if aspect_ratio > 1:  # 宽度比高度大
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:  # 高度比宽度大或相等
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    # 放缩图像但保持不压缩
    resized_leaf = cv2.resize(cropped_leaf, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # 创建一个全黑色的背景图像
    final_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # 将缩放后的叶片图像放置在背景图像的中央
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    final_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_leaf

    return final_image


def process_images_in_folder(input_folder, output_folder, output_size=(384, 384)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            output_image_path = os.path.join(output_folder, filename)

            # 提取叶片并保存结果
            final_image = extract_leaf(image_path, output_size)
            cv2.imwrite(output_image_path, final_image)

    print("Processing complete!")


# 使用示例
input_folder = "D:\\ic\\rotate\\disease"
output_folder = "D:\\ic\\rotate\\disease3"

process_images_in_folder(input_folder, output_folder)



