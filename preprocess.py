import cv2
import numpy as np
import os
from pathlib import Path  # 使用 pathlib 来更方便地处理路径


def crop_container(image):
    """
    根据提供的逻辑裁剪图像中的最大轮廓（集装箱）。

    注意：我对你的原始代码做了修正。
    1.  你原来的裁剪 `image[y-10:y+h+10, x-10:x+w+10]` 在 x 或 y 接近 0 时会产生负索引，
        这在 numpy 切片中意味着“从末尾倒数”，会导致裁剪错误。
    2.  你原来的 `np.clip(cropped, 0, 255)` 是用来裁剪像素的 *值*（颜色），
        而不是裁剪 *坐标*。

    我已经修正了这个问题，使用 max(0, ...) 和 min(width, ...) 来确保坐标总是在图像边界内。
    """

    # 0. 获取原始图像尺寸，用于安全裁剪
    h_img, w_img = image.shape[:2]

    # 1. 转灰度图，方便边缘检测
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. 去噪（高斯模糊），避免把污渍当边缘
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. 边缘检测（Canny），调阈值找集装箱的硬边缘
    edges = cv2.Canny(blur, 50, 150)

    # 4. 找轮廓
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("Warning: 未找到轮廓，将返回原图。")
        return image  # 没找到轮廓就返回原图

    # 5. 只留面积最大的（大概率是集装箱）
    max_contour = max(contours, key=cv2.contourArea)

    # 6. 给轮廓画个 bounding box
    x, y, w, h = cv2.boundingRect(max_contour)

    # 7. 计算带 10 像素边距的安全裁剪坐标 [修正点]
    x1 = max(0, x - 10)
    y1 = max(0, y - 10)
    x2 = min(w_img, x + w + 10)  # 确保不超过原始宽度
    y2 = min(h_img, y + h + 10)  # 确保不超过原始高度

    # 8. 裁剪集装箱区域
    cropped = image[y1:y2, x1:x2]

    # 9. 原始的 np.clip 已不需要，因为坐标在第 7 步已保证安全

    return cropped


def process_images_in_folder(input_folder, output_folder):
    """
    遍历输入文件夹中的所有图片，应用 crop_container 函数，
    并将结果保存到输出文件夹。
    """

    # 1. 定义路径
    input_dir = Path(input_folder)
    output_dir = Path(output_folder)

    # 2. 创建输出目录（如果它不存在的话）
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"开始处理图片...")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")

    # 3. 定义支持的图片格式
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]

    # 4. 获取所有图片文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_dir.glob(ext))

    if not image_files:
        print(f"错误: 在 {input_dir} 中未找到任何图片文件。")
        return

    print(f"总共找到了 {len(image_files)} 张图片。")

    # 5. 遍历、处理和保存
    processed_count = 0
    failed_count = 0
    for img_path in image_files:
        try:
            # 读取图片 (使用 str() 来兼容 cv2)
            # 使用 cv2.IMREAD_COLOR 确保始终读取 3 通道 BGR 图像
            image = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)

            if image is None:
                print(f"Warning: 无法读取图片 {img_path.name}，已跳过。")
                failed_count += 1
                continue

            # 应用你的裁剪函数
            cropped_image = crop_container(image)

            # 定义输出路径
            output_path = output_dir / img_path.name

            # 保存图片 (使用 str() 来兼容 cv2)
            # 使用 imencode 来处理可能包含中文的路径
            is_success, buffer = cv2.imencode(img_path.suffix, cropped_image)
            if is_success:
                with open(str(output_path), 'wb') as f:
                    f.write(buffer)
                processed_count += 1
            else:
                print(f"Warning: 无法编码图片 {img_path.name}，已跳过。")
                failed_count += 1

        except Exception as e:
            print(f"处理图片 {img_path.name} 时发生严重错误: {e}")
            failed_count += 1

    print("\n--- 处理完成 ---")
    print(f"成功处理并保存: {processed_count} 张")
    print(f"失败或跳过: {failed_count} 张")
    print(f"裁剪后的图片已保存至: {output_dir}")


# --- 主程序入口 ---
if __name__ == "__main__":
    # 1. 指定你的原始图片文件夹
    # (r"..." 是原始字符串，可以防止 \ 被转义)
    INPUT_DIR = r"D:\CODE\GithubCode\Yolo\ultralytics-8.3.163\datasets\container\images\train"

    # 2. 自动设置输出文件夹
    # (例如: D:\...\images\train_cropped)
    OUTPUT_DIR = Path(INPUT_DIR).parent / (Path(INPUT_DIR).name + "_cropped")

    # 3. 运行处理函数
    process_images_in_folder(INPUT_DIR, str(OUTPUT_DIR))