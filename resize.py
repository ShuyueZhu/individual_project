import cv2
import os
import numpy as np


def crop_and_resize_image(image, size=(384, 384)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(thresh)
    cropped_image = image[y:y + h, x:x + w]

    aspect_ratio = w / h
    target_width, target_height = size

    if aspect_ratio > 1:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    resized_image = cv2.resize(cropped_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    final_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    final_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image

    return final_image


def process_images_in_folder(input_folder, output_folder, size=(384, 384), quality=95):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            processed_image = crop_and_resize_image(image, size)

            output_path = os.path.join(output_folder, filename)

            # 判断文件格式并设置保存参数
            if filename.endswith(".jpg"):
                cv2.imwrite(output_path, processed_image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            else:
                cv2.imwrite(output_path, processed_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    print("Processing complete!")



# 使用示例
input_folder = "D:\\ic\\rotate\\disease"
output_folder = "D:\\ic\\rotate\\disease3"
process_images_in_folder(input_folder, output_folder, quality=100)
