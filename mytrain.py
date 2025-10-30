import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import albumentations as A
from ultralytics import YOLO

# -------------------------------
# 1️⃣ 配置增强参数（Z增强）
# -------------------------------
p_motionblur = 0.3
p_gaussnoise = 0.3
p_brightness = 0.5
p_clahe = 0.5
p_rgbshift = 0.5
p_flip = 0.5


def get_z_augmentations():
    return A.Compose([
        A.MotionBlur(blur_limit=(3, 7), p=p_motionblur),
        A.GaussNoise(p=p_gaussnoise),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=p_brightness),
        A.CLAHE(clip_limit=(1.0, 4.0), tile_grid_size=(8, 8), p=p_clahe),
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=p_rgbshift),
        A.HorizontalFlip(p=p_flip)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.1))


# -------------------------------
# 2️⃣ 可视化训练前增强效果
# -------------------------------
def visualize_augmentations(image_path, bboxes, labels, n_samples=5):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    aug = get_z_augmentations()

    plt.figure(figsize=(15, 5))
    for i in range(n_samples):
        augmented = aug(image=image.copy(), bboxes=bboxes.copy(), class_labels=labels.copy())
        img_aug = augmented['image']
        bboxes_aug = augmented['bboxes']

        for bbox in bboxes_aug:
            x_center, y_center, w, h = bbox
            h_img, w_img, _ = img_aug.shape
            x1 = int((x_center - w / 2) * w_img)
            y1 = int((y_center - h / 2) * h_img)
            x2 = int((x_center + w / 2) * w_img)
            y2 = int((y_center + h / 2) * h_img)
            cv2.rectangle(img_aug, (x1, y1), (x2, y2), (255, 0, 0), 2)

        plt.subplot(1, n_samples, i + 1)
        plt.imshow(img_aug)
        plt.axis('off')
    plt.show()


# -------------------------------
# 3️⃣ 加载示例标签（YOLO格式）
# -------------------------------
def load_yolo_labels(image_path):
    label_path = image_path.replace('images', 'labels').replace('.jpg', '.txt')
    bboxes = []
    class_labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                cls = int(parts[0])
                bbox = list(map(float, parts[1:5]))  # x_center y_center w h
                class_labels.append(cls)
                bboxes.append(bbox)
    return bboxes, class_labels


# -------------------------------
# 4️⃣ Windows 多进程保护
# -------------------------------
if __name__ == '__main__':
    # 示例图片路径
    sample_img_path = r'D:\CODE\GithubCode\Yolo\ultralytics-8.3.163\datasets\container\images\train\1.jpg'
    sample_bboxes, sample_labels = load_yolo_labels(sample_img_path)

    # 可视化增强效果
    visualize_augmentations(sample_img_path, sample_bboxes, sample_labels, n_samples=5)

    # 初始化YOLO模型
    model = YOLO('yolo11n.pt')  # 替换为你的YOLOv11权重

    # 开始训练
    model.train(
        data=r'D:\CODE\GithubCode\Yolo\ultralytics-8.3.163\ultralytics\cfg\datasets\container.yaml',
        epochs=100,
        batch=16,
        imgsz=640,
        name='train_z_augment',
        augment=True,  # YOLO内置增强
        workers=0  # Windows 推荐先用0，多进程容易报错
    )
