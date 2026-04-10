# train_au_casme_improved.py
import os
import cv2
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import datetime

# -----------------------------
# 配置
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16  # 增大batch size
EPOCHS = 150
LR = 1e-3  # 提高学习率
IMG_SIZE = 224
PATIENCE = 30
MIN_DELTA = 0.005
USE_FOCAL_LOSS = True  # 使用Focal Loss
MIN_AU_COUNT = 5  # 最小出现次数

SAVE_DIR = "./casme_au_models_improved"
os.makedirs(SAVE_DIR, exist_ok=True)

# CASME2 AU列表（保持不变）
CASME_AU_LIST = [
    "1", "2", "4", "5", "6", "7", "9", "10", "12", "14",
    "15", "16", "17", "18", "20", "24", "25", "26", "38",
    "L1", "L2", "L4", "L6", "L7", "L9", "L10", "L12", "L14", "L20",
    "R1", "R2", "R4", "R10", "R12", "R14"
]


# -----------------------------
# Focal Loss
# -----------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()


# -----------------------------
# 数据集类（改进版）
# -----------------------------
class CASMEAUMultiLabelDataset(Dataset):
    def __init__(self, root_dir, transform=None, selected_au_indices=None):
        self.samples = []
        self.transform = transform
        self.selected_au_indices = selected_au_indices

        print(f"正在加载CASME2 AU多标签数据集: {root_dir}")

        casme_path = os.path.join(root_dir, "csame")
        if not os.path.exists(casme_path):
            print(f"警告: CASME2 目录不存在: {casme_path}")
            return

        # 统计AU出现次数
        au_counts = np.zeros(len(CASME_AU_LIST))

        for subject in os.listdir(casme_path):
            subject_path = os.path.join(casme_path, subject)
            if not os.path.isdir(subject_path):
                continue

            for video in os.listdir(subject_path):
                video_path = os.path.join(subject_path, video)
                if not os.path.isdir(video_path):
                    continue

                json_file = None
                for f in os.listdir(video_path):
                    if f.endswith('.json'):
                        json_file = f
                        break

                if not json_file:
                    continue

                apex_path = os.path.join(video_path, "apex.jpg")
                if not os.path.exists(apex_path):
                    continue

                try:
                    json_path = os.path.join(video_path, json_file)
                    with open(json_path, "r") as f:
                        info = json.load(f)
                        au_str = info.get("au", "")

                        if au_str == "":
                            continue

                        au_labels = au_str.split("+") if "+" in au_str else [au_str]

                        # 创建multi-hot向量
                        multi_hot = [0] * len(CASME_AU_LIST)
                        valid_au_found = False

                        for au in au_labels:
                            au = au.strip()
                            if au in CASME_AU_LIST:
                                idx = CASME_AU_LIST.index(au)
                                multi_hot[idx] = 1
                                au_counts[idx] += 1
                                valid_au_found = True

                        if valid_au_found:
                            self.samples.append((apex_path, multi_hot))

                except Exception as e:
                    continue

        # 筛选高频AU
        if selected_au_indices is None:
            selected_au_indices = [i for i, count in enumerate(au_counts) if count >= MIN_AU_COUNT]

        self.selected_au_indices = selected_au_indices
        self.selected_au_list = [CASME_AU_LIST[i] for i in selected_au_indices]

        # 过滤样本，只保留有筛选后AU的样本
        filtered_samples = []
        for img_path, multi_hot in self.samples:
            filtered_multi_hot = [multi_hot[i] for i in selected_au_indices]
            if sum(filtered_multi_hot) > 0:  # 至少有一个筛选后的AU
                filtered_samples.append((img_path, filtered_multi_hot))

        self.samples = filtered_samples

        print(f"\n{'=' * 60}")
        print(f"CASME2 AU多标签数据集加载完成！")
        print(f"原始AU数量: {len(CASME_AU_LIST)}")
        print(f"筛选后AU数量: {len(self.selected_au_list)}")
        print(f"筛选后AU: {self.selected_au_list}")
        print(f"总样本数: {len(self.samples)}")

        # 打印筛选后的AU分布
        if len(self.samples) > 0:
            selected_counts = np.zeros(len(self.selected_au_list))
            for _, multi_hot in self.samples:
                selected_counts += np.array(multi_hot)

            print(f"\n筛选后AU分布:")
            for i, (au, count) in enumerate(zip(self.selected_au_list, selected_counts)):
                if count > 0:
                    print(f"  {au}: {int(count)} 个样本 ({count / len(self.samples) * 100:.1f}%)")

        print(f"{'=' * 60}")

        if len(self.samples) == 0:
            raise ValueError("❌ CASME2 数据集为空！")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, multi_hot = self.samples[idx]

        try:
            img = cv2.imread(img_path)
            if img is None:
                random_idx = np.random.randint(0, len(self.samples))
                return self.__getitem__(random_idx)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.transform:
                img = self.transform(img)

            return img, torch.tensor(multi_hot, dtype=torch.float32)

        except Exception as e:
            random_idx = np.random.randint(0, len(self.samples))
            return self.__getitem__(random_idx)


# -----------------------------
# 数据增强/预处理
# -----------------------------
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
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

print("=" * 60)
print("开始加载CASME2 AU多标签数据集...")
print("=" * 60)

try:
    full_dataset = CASMEAUMultiLabelDataset(data_dir, transform=train_transform)
except ValueError as e:
    print(f"错误: {e}")
    exit(1)

# 划分训练集和验证集
from sklearn.model_selection import train_test_split

train_indices, val_indices = train_test_split(
    range(len(full_dataset)),
    test_size=0.2,
    random_state=42
)

train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

val_dataset.dataset.transform = val_transform

print(f"\n训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


# -----------------------------
# 模型
# -----------------------------
class CASMEAUPredictionModel(nn.Module):
    def __init__(self, num_au, dropout_rate=0.5):
        super(CASMEAUPredictionModel, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_au)
        )

    def forward(self, x):
        return self.backbone(x)


num_au = len(full_dataset.selected_au_list)
model = CASMEAUPredictionModel(num_au=num_au, dropout_rate=0.5)
model = model.to(device)

# 损失函数
if USE_FOCAL_LOSS:
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    print("使用 Focal Loss")
else:
    # 计算类别权重
    pos_counts = np.zeros(num_au)
    for _, label in train_dataset:
        pos_counts += label.numpy()

    pos_ratio = pos_counts / len(train_dataset)
    pos_weights = 1.0 / (pos_ratio + 1e-6)
    pos_weights = pos_weights / pos_weights.mean()
    pos_weights = torch.tensor(pos_weights, dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    print("使用加权 BCE Loss")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)


# -----------------------------
# 训练函数
# -----------------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    all_preds = []
    all_labels = []

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
        preds = torch.sigmoid(outputs) > 0.5
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # 计算当前batch的F1
        if len(all_preds) > 0:
            batch_f1 = f1_score(all_labels[-len(labels):], all_preds[-len(labels):],
                                average='macro', zero_division=0)
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1), 'f1': batch_f1})

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    sample_acc = (all_preds == all_labels).all(axis=1).mean()
    au_acc = (all_preds == all_labels).mean()
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return running_loss / len(loader), sample_acc, au_acc, f1_macro


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Validation"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    sample_acc = (all_preds == all_labels).all(axis=1).mean()
    au_acc = (all_preds == all_labels).mean()
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # 计算每个AU的F1
    au_f1 = {}
    for i, au_name in enumerate(full_dataset.selected_au_list):
        if np.sum(all_labels[:, i]) > 0:
            au_f1_i = f1_score(all_labels[:, i], all_preds[:, i], zero_division=0)
            au_f1[au_name] = au_f1_i

    return running_loss / len(loader), sample_acc, au_acc, f1_macro, au_f1


def save_model(model, optimizer, epoch, metrics, is_best=False):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'au_list': full_dataset.selected_au_list
    }

    if is_best:
        best_model_path = os.path.join(SAVE_DIR, "best_casme_au_model.pth")
        torch.save(checkpoint, best_model_path)
        print(f"✅ 最佳CASME2模型已保存: {best_model_path}")

    if (epoch + 1) % 10 == 0:
        checkpoint_path = os.path.join(SAVE_DIR, f"casme_au_model_epoch_{epoch + 1}_{timestamp}.pth")
        torch.save(checkpoint, checkpoint_path)

    latest_path = os.path.join(SAVE_DIR, "latest_casme_au_model.pth")
    torch.save(checkpoint, latest_path)


# -----------------------------
# 主训练循环
# -----------------------------
print("\n" + "=" * 60)
print("开始训练CASME2 AU多标签模型（改进版）")
print("=" * 60)
print(f"设备: {device}")
print(f"AU数量: {num_au}")
print(f"训练样本数: {len(train_dataset)}")
print(f"验证样本数: {len(val_dataset)}")
print(f"使用Focal Loss: {USE_FOCAL_LOSS}")
print("=" * 60)

best_val_f1 = 0
patience_counter = 0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print("-" * 40)

    train_loss, train_sample_acc, train_au_acc, train_f1 = train_epoch(
        model, train_loader, criterion, optimizer, device
    )

    val_loss, val_sample_acc, val_au_acc, val_f1, au_f1 = validate(
        model, val_loader, criterion, device
    )

    # 更新学习率
    scheduler.step(val_f1)
    current_lr = optimizer.param_groups[0]['lr']

    print(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
    print(f"Learning Rate: {current_lr:.6f}")

    # 打印Top 5 AU的F1分数
    if au_f1:
        sorted_au = sorted(au_f1.items(), key=lambda x: x[1], reverse=True)[:5]
        print("Top 5 AU F1 Score:")
        for au_name, f1 in sorted_au:
            if f1 > 0:
                print(f"  {au_name}: {f1:.4f}")

    # 早停判断
    if val_f1 > best_val_f1 + MIN_DELTA:
        best_val_f1 = val_f1
        patience_counter = 0
        metrics = {'val_f1': val_f1, 'val_au_acc': val_au_acc, 'epoch': epoch}
        save_model(model, optimizer, epoch, metrics, is_best=True)
        print(f"✅ 新的最佳CASME2模型！F1: {val_f1:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n早停触发！最佳F1: {best_val_f1:.4f}")
            break

print("\n" + "=" * 60)
print("CASME2训练完成！")
print("=" * 60)
print(f"最佳验证F1分数: {best_val_f1:.4f}")