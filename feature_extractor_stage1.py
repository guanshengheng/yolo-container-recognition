import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import pandas as pd
from collections import Counter

# =======================
# 配置参数
# =======================
train_img_dir = r"D:\CODE\GithubCode\Yolo\ultralytics-8.3.163\datasets\container\images\train"
train_label_dir = r"D:\CODE\GithubCode\Yolo\ultralytics-8.3.163\datasets\container\labels\train"
val_img_dir = r"D:\CODE\GithubCode\Yolo\ultralytics-8.3.163\datasets\container\images\val"
val_label_dir = r"D:\CODE\GithubCode\Yolo\ultralytics-8.3.163\datasets\container\labels\val"
test_img_dir = r"D:\CODE\GithubCode\Yolo\ultralytics-8.3.163\datasets\container\images\test"
test_label_dir = r"D:\CODE\GithubCode\Yolo\ultralytics-8.3.163\datasets\container\labels\test"

batch_size = 16
num_epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 3  # Dent=0, Hole=1, Rusty=2

# 保存特征文件
feature_save_train = "train_features.pt"
feature_save_val = "val_features.pt"
feature_save_test = "test_features.pt"

# =======================
# 工具函数
# =======================
def find_images_and_labels(img_dir, label_dir=None):
    images = []
    labels = []
    for root, _, files in os.walk(img_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                img_path = os.path.join(root, file)
                images.append(img_path)

                if label_dir:
                    label_file = os.path.join(label_dir, os.path.splitext(file)[0] + ".txt")
                    if os.path.exists(label_file):
                        with open(label_file, 'r') as f:
                            lines = f.readlines()
                            label = int(lines[0].strip().split()[0])
                    else:
                        label = -1
                    labels.append(label)
    if label_dir:
        return images, labels
    else:
        return images

# =======================
# Dataset
# =======================
class YoloDataset(Dataset):
    def __init__(self, img_paths, labels=None, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.labels is not None:
            label = self.labels[idx]
            return img, label
        else:
            return img

# =======================
# 数据增强和预处理
# =======================
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

transform_eval = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# =======================
# 加载 train/val/test 数据
# =======================
train_images, train_labels = find_images_and_labels(train_img_dir, train_label_dir)
val_images, val_labels = find_images_and_labels(val_img_dir, val_label_dir)
test_images, test_labels = find_images_and_labels(test_img_dir, test_label_dir)

print(f"训练集: {len(train_images)} 张")
print(f"验证集: {len(val_images)} 张")
print(f"测试集: {len(test_images)} 张")

train_dataset = YoloDataset(train_images, train_labels, transform=transform_train)
val_dataset = YoloDataset(val_images, val_labels, transform=transform_eval)
test_dataset = YoloDataset(test_images, test_labels, transform=transform_eval)

# 类别平衡采样
label_counts = Counter(train_labels)
weights = [1.0 / label_counts[label] for label in train_labels]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# =======================
# 构建模型
# =======================
# =======================
# 构建模型（修改版）
# =======================
print("正在初始化 ResNet18 模型...")

from torchvision.models import resnet18, ResNet18_Weights

# 尝试优先加载本地权重
weights_path = r"C:\Users\GSH\.cache\torch\hub\checkpoints\resnet18-f37072fd.pth"

if os.path.exists(weights_path):
    print(f"检测到本地权重文件: {weights_path}")
    state_dict = torch.load(weights_path, map_location="cpu")
    model = resnet18(weights=None)  # 不再联网下载
    model.load_state_dict(state_dict)
else:
    print("⚠️ 未检测到本地权重文件，将不加载预训练参数（weights=None）")
    print("（如需使用预训练，请手动下载：https://download.pytorch.org/models/resnet18-f37072fd.pth）")
    model = resnet18(weights=None)

# 修改最后一层分类头
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# =======================
# 训练模型
# =======================
print("开始训练模型...")
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# =======================
# 特征提取函数
# =======================
def extract_features(dataloader, model, save_path):
    model.eval()
    features = []
    with torch.no_grad():
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            x = model.conv1(imgs)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            x = model.avgpool(x)
            x = torch.flatten(x, 1)
            features.append(x.cpu())
    features = torch.cat(features, dim=0)
    torch.save(features, save_path)
    print(f"Features saved to {save_path}")

# =======================
# 提取 train/val/test 特征
# =======================
print("提取 train 特征...")
extract_features(train_loader, model, feature_save_train)

print("提取 val 特征并预测...")
model.eval()
val_features, val_preds = [], []
with torch.no_grad():
    for imgs, _ in val_loader:
        imgs = imgs.to(device)
        x = model.conv1(imgs)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x_feat = model.avgpool(x)
        x_feat = torch.flatten(x_feat, 1)
        val_features.append(x_feat.cpu())
        output = model(imgs)
        pred = torch.argmax(output, dim=1).item()
        val_preds.append(pred)
val_features = torch.cat(val_features, dim=0)
torch.save(val_features, feature_save_val)

cls_names = ["Dent", "Hole", "Rusty"]
val_results = {"path": val_images, "pred_class": [cls_names[p] for p in val_preds]}
pd.DataFrame(val_results).to_csv("val_predictions.csv", index=False)
print("Val predictions saved to val_predictions.csv")

print("提取 test 特征并预测...")
test_features, test_preds = [], []
with torch.no_grad():
    for imgs, _ in test_loader:
        imgs = imgs.to(device)
        x = model.conv1(imgs)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x_feat = model.avgpool(x)
        x_feat = torch.flatten(x_feat, 1)
        test_features.append(x_feat.cpu())
        output = model(imgs)
        pred = torch.argmax(output, dim=1).item()
        test_preds.append(pred)
test_features = torch.cat(test_features, dim=0)
torch.save(test_features, feature_save_test)

test_results = {"path": test_images, "pred_class": [cls_names[p] for p in test_preds]}
pd.DataFrame(test_results).to_csv("test_predictions.csv", index=False)
print("Test predictions saved to test_predictions.csv")

print("全部完成 ✅")
