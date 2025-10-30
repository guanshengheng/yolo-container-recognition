import os
import shutil
import random
from glob import glob

# --- 1. 配置您的路径 ---

# 指向您 'container' 数据集的根目录 (根据您的截图)
CONTAINER_DATASET_DIR = r"D:\CODE\GithubCode\Yolo\ultralytics-8.3.163\datasets\container"

# 验证集所占的比例 (例如 0.15 表示 15%)
VAL_SPLIT_RATIO = 0.15

# --- 2. 定义所有路径 ---
# 源目录 (现有的训练集)
src_images_dir = os.path.join(CONTAINER_DATASET_DIR, "images", "train")
src_labels_dir = os.path.join(CONTAINER_DATASET_DIR, "labels", "train")

# 目标目录 (需要创建的验证集)
dest_images_dir = os.path.join(CONTAINER_DATASET_DIR, "images", "val")
dest_labels_dir = os.path.join(CONTAINER_DATASET_DIR, "labels", "val")


def create_val_dirs():
    """创建验证集目录 (images/val 和 labels/val)"""
    os.makedirs(dest_images_dir, exist_ok=True)
    os.makedirs(dest_labels_dir, exist_ok=True)
    print(f"创建或确认目录: {dest_images_dir}")
    print(f"创建或确认目录: {dest_labels_dir}")


def split_train_to_val():
    """
    执行划分。
    从 'train' 目录中抓取文件，并 '移动' 它们到 'val' 目录。
    """

    # 查找所有图片文件 (假设是 .jpg, .png, .jpeg)
    image_files = glob(os.path.join(src_images_dir, "*.jpg"))
    image_files.extend(glob(os.path.join(src_images_dir, "*.png")))
    image_files.extend(glob(os.path.join(src_images_dir, "*.jpeg")))

    if not image_files:
        print(f"错误：在 '{src_images_dir}' 中未找到任何图片文件。请检查路径。")
        print("脚本已停止。")
        return

    # 随机打乱列表
    random.shuffle(image_files)

    # 计算需要移动到验证集的文件数量
    total_images = len(image_files)
    num_val = int(total_images * VAL_SPLIT_RATIO)
    num_train = total_images - num_val

    if num_val == 0:
        print(f"总图片数 ({total_images}) 太少，无法按比例 {VAL_SPLIT_RATIO} 划分出验证集。")
        print("脚本已停止。请检查图片数量或调整比例。")
        return

    # 选出要移动的验证集文件
    val_files_to_move = image_files[:num_val]

    print(f"总共找到 {total_images} 张原始训练图片。")
    print(f"划分比例: {VAL_SPLIT_RATIO * 100:.0f}% 将成为验证集。")
    print(f"将移动 {num_val} 张图片到 'val' 目录。")
    print(f"剩余 {num_train} 张图片保留在 'train' 目录。")

    # --- 3. 移动文件 ---
    moved_imgs = 0
    moved_lbls = 0

    for img_src_path in val_files_to_move:
        try:
            # 获取文件名 (例如: "image1.jpg")
            img_filename = os.path.basename(img_src_path)

            # 定义标签文件名 (例如: "image1.txt")
            label_filename = os.path.splitext(img_filename)[0] + ".txt"

            # 定义标签的源路径和目标路径
            label_src_path = os.path.join(src_labels_dir, label_filename)
            img_dest_path = os.path.join(dest_images_dir, img_filename)
            label_dest_path = os.path.join(dest_labels_dir, label_filename)

            # 移动图片
            shutil.move(img_src_path, img_dest_path)
            moved_imgs += 1

            # 检查对应的标签文件是否存在，存在才移动
            if os.path.exists(label_src_path):
                shutil.move(label_src_path, label_dest_path)
                moved_lbls += 1

        except Exception as e:
            print(f"处理文件 {img_src_path} 时出错: {e}")

    print("\n--- 文件移动完成 ---")
    print(f"总共移动了 {moved_imgs} 张图片到 'images/val'")
    print(f"总共移动了 {moved_lbls} 个标签到 'labels/val'")
    print(f"（{moved_imgs - moved_lbls} 张图片没有对应的标签文件，这可能是无残损样本，是正常的。）")
    print("\n数据集划分完毕！")


# --- 运行脚本 ---
if __name__ == "__main__":
    print("开始执行数据集划分脚本...")
    create_val_dirs()
    split_train_to_val()