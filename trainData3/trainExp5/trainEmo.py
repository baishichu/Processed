import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
import json
import numpy as np

# -----------------------------
# 配置
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-4
IMG_SIZE = 224
PATIENCE = 10

# 微表情类别（根据你的数据集调整）
EMO_CLASSES = ["disgust", "fear", "happiness", "others", "repression", "sadness", "surprise","Anger","Contempt","Disgust","Fear","Happiness","Other","Sadness","Surprise"]


# -----------------------------
# 修复后的数据集类
# -----------------------------
class MicroExpressionDataset(Dataset):
    """
    修复版微表情数据集
    支持 CASME2 和 SAMM 的目录结构
    """

    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        self.class_counts = {cls: 0 for cls in EMO_CLASSES}

        print(f"正在加载数据集: {root_dir}")

        # 遍历数据目录
        for dataset_type in os.listdir(root_dir):
            dataset_path = os.path.join(root_dir, dataset_type)
            if not os.path.isdir(dataset_path):
                continue

            print(f"\n处理数据集: {dataset_type}")

            # 遍历被试者
            for subject in os.listdir(dataset_path):
                subject_path = os.path.join(dataset_path, subject)
                if not os.path.isdir(subject_path):
                    continue

                # 遍历视频/样本
                for video in os.listdir(subject_path):
                    video_path = os.path.join(subject_path, video)
                    if not os.path.isdir(video_path):
                        continue

                    # 查找JSON文件
                    json_file = None
                    for f in os.listdir(video_path):
                        if f.endswith('.json'):
                            json_file = f
                            break

                    if not json_file:
                        print(f"  警告: 在 {video_path} 中未找到JSON文件")
                        continue

                    # 查找峰值帧
                    apex_path = os.path.join(video_path, "apex.jpg")
                    if not os.path.exists(apex_path):
                        print(f"  警告: 在 {video_path} 中未找到 apex.jpg")
                        continue

                    # 读取JSON文件
                    try:
                        json_path = os.path.join(video_path, json_file)
                        with open(json_path, "r") as f:
                            info = json.load(f)
                            emo_label = info.get("emo", None)

                            # 检查情绪标签是否有效
                            if emo_label is None:
                                print(f"  警告: {json_file} 中没有 'emo' 字段")
                                continue

                            if emo_label not in EMO_CLASSES:
                                print(f"  警告: 未知情绪标签 '{emo_label}' in {json_file}")
                                continue

                            # 添加到样本列表
                            label_idx = EMO_CLASSES.index(emo_label)
                            self.samples.append((apex_path, label_idx))
                            self.class_counts[emo_label] += 1

                    except Exception as e:
                        print(f"  错误: 读取 {json_file} 失败: {e}")
                        continue

        # 打印统计信息
        print(f"\n{'=' * 50}")
        print(f"数据集加载完成！")
        print(f"总样本数: {len(self.samples)}")
        print(f"类别分布:")
        for cls, count in self.class_counts.items():
            if count > 0:
                print(f"  {cls}: {count} 个样本 ({count / len(self.samples) * 100:.1f}%)")
        print(f"{'=' * 50}")

        if len(self.samples) == 0:
            raise ValueError("❌ 数据集为空！请检查数据路径和文件格式。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"错误: 无法读取图像 {img_path}")
                # 返回随机样本
                random_idx = np.random.randint(0, len(self.samples))
                return self.__getitem__(random_idx)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.transform:
                img = self.transform(img)

            return img, label

        except Exception as e:
            print(f"错误: 处理图像失败 {img_path}: {e}")
            random_idx = np.random.randint(0, len(self.samples))
            return self.__getitem__(random_idx)


# -----------------------------
# 数据增强/预处理
# -----------------------------
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----------------------------
# 数据加载
# -----------------------------
data_dir = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/trainData3/data3"

# 加载完整数据集
print("=" * 50)
print("开始加载数据集...")
print("=" * 50)

try:
    full_dataset = MicroExpressionDataset(data_dir, transform=train_transform)
except ValueError as e:
    print(f"错误: {e}")
    exit(1)

# 划分训练集和验证集
from sklearn.model_selection import train_test_split

# 获取所有标签
labels = [sample[1] for sample in full_dataset.samples]
train_indices, val_indices = train_test_split(
    range(len(full_dataset)),
    test_size=0.2,
    stratify=labels,
    random_state=42
)

# 创建子集
train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

# 验证集使用不同的transform
val_dataset.dataset.transform = val_transform

print(f"\n训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


# -----------------------------
# 模型定义
# -----------------------------
class EmotionResNet(nn.Module):
    def __init__(self, num_classes=7, dropout_rate=0.3):
        super(EmotionResNet, self).__init__()

        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# 创建模型
model = EmotionResNet(num_classes=len(EMO_CLASSES), dropout_rate=0.3)
model = model.to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


# -----------------------------
# 训练函数
# -----------------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1),
            'acc': correct / total
        })

    return running_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Validation"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / len(loader), correct / total


# -----------------------------
# 训练循环
# -----------------------------
print("\n" + "=" * 50)
print("开始训练")
print("=" * 50)
print(f"设备: {device}")
print(f"类别数: {len(EMO_CLASSES)}")
print(f"训练样本数: {len(train_dataset)}")
print(f"验证样本数: {len(val_dataset)}")
print(f"批次大小: {BATCH_SIZE}")
print(f"学习率: {LR}")
print("=" * 50)

best_val_acc = 0
patience_counter = 0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print("-" * 40)

    # 训练
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

    # 验证
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    # 更新学习率
    scheduler.step()

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
            'class_names': EMO_CLASSES
        }, "best_emotion_model.pth")
        print(f"✅ 保存最佳模型 (Val Acc: {val_acc:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n早停触发！最佳验证准确率: {best_val_acc:.4f}")
            break

print("\n" + "=" * 50)
print(f"训练完成！最佳验证准确率: {best_val_acc:.4f}")
print("=" * 50)