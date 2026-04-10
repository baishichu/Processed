# unet_expression_regions.py
import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

# -----------------------------
# 配置
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-4
IMG_SIZE = 224
SAVE_DIR = "./unet_expression_models"
os.makedirs(SAVE_DIR, exist_ok=True)

# 数据目录（请根据实际情况修改）
DATA_DIR = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/trainData3/data3"


# -----------------------------
# 1. 数据集（apex → onset）
# -----------------------------
class ApexOnsetDataset(Dataset):
    def __init__(self, root_dir, transform=None, augment=False):
        self.samples = []
        self.transform = transform
        self.augment = augment

        print(f"正在加载Apex-Onset配对数据集: {root_dir}")

        # 支持多个数据集
        for dataset in ["csame", "SAMM"]:
            dataset_path = os.path.join(root_dir, dataset)
            if not os.path.exists(dataset_path):
                continue

            print(f"处理数据集: {dataset}")

            for subject in os.listdir(dataset_path):
                subject_path = os.path.join(dataset_path, subject)
                if not os.path.isdir(subject_path):
                    continue

                for video in os.listdir(subject_path):
                    video_path = os.path.join(subject_path, video)
                    if not os.path.isdir(video_path):
                        continue

                    apex = os.path.join(video_path, "apex.jpg")
                    onset = os.path.join(video_path, "onset.jpg")

                    if os.path.exists(apex) and os.path.exists(onset):
                        self.samples.append((apex, onset))

        print(f"✅ U-Net数据量: {len(self.samples)}")

        if len(self.samples) == 0:
            raise ValueError("未找到任何Apex-Onset配对数据！")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        apex_path, onset_path = self.samples[idx]

        apex = cv2.imread(apex_path)
        onset = cv2.imread(onset_path)

        if apex is None or onset is None:
            return self.__getitem__((idx + 1) % len(self.samples))

        apex = cv2.cvtColor(apex, cv2.COLOR_BGR2RGB)
        onset = cv2.cvtColor(onset, cv2.COLOR_BGR2RGB)

        # 数据增强
        if self.augment and np.random.random() > 0.5:
            # 水平翻转
            apex = cv2.flip(apex, 1)
            onset = cv2.flip(onset, 1)

        if self.transform:
            apex = self.transform(apex)
            onset = self.transform(onset)

        return apex, onset


# -----------------------------
# 2. U-Net模型（改进版）
# -----------------------------
class ImprovedUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super(ImprovedUNet, self).__init__()

        # Encoder
        self.enc1 = self._block(in_channels, features[0])
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = self._block(features[0], features[1])
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = self._block(features[1], features[2])
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = self._block(features[2], features[3])
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self._block(features[3], features[3] * 2)

        # Decoder
        self.up4 = nn.ConvTranspose2d(features[3] * 2, features[3], kernel_size=2, stride=2)
        self.dec4 = self._block(features[3] * 2, features[3])

        self.up3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.dec3 = self._block(features[2] * 2, features[2])

        self.up2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.dec2 = self._block(features[1] * 2, features[1])

        self.up1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.dec1 = self._block(features[0] * 2, features[0])

        self.out = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)


# -----------------------------
# 3. Grad-CAM（改进版）
# -----------------------------
class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None

        # 默认使用decoder的最后一层
        if target_layer is None:
            target_layer = model.enc4[-1] # ReLU层

        self.target_layer = target_layer
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, x):
        self.model.zero_grad()

        # 前向传播
        output = self.model(x)

        # 使用输出图像的差异作为损失
        loss = torch.abs(output - x).mean()
        loss.backward()

        # 计算权重和热力图
        grads = self.gradients
        acts = self.activations

        if grads is None or acts is None:
            raise ValueError("梯度或激活值为None")

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        # 归一化
        cam = cam.squeeze().cpu().detach().numpy()
        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        return cam


# -----------------------------
# 4. 面部区域定义
# -----------------------------
class FaceRegionAnalyzer:
    def __init__(self):
        # 定义更详细的面部区域
        self.regions = {
            'left_eyebrow': (0, 80, 0, 112),  # (y1, y2, x1, x2)
            'right_eyebrow': (0, 80, 112, 224),
            'left_eye': (80, 130, 0, 112),
            'right_eye': (80, 130, 112, 224),
            'nose': (130, 170, 70, 154),
            'left_cheek': (130, 180, 0, 80),
            'right_cheek': (130, 180, 144, 224),
            'upper_lip': (170, 190, 70, 154),
            'lower_lip': (190, 210, 70, 154),
            'left_mouth_corner': (175, 195, 50, 80),
            'right_mouth_corner': (175, 195, 144, 174),
            'chin': (210, 224, 70, 154)
        }

    threshold = 0.1
    def get_active_regions(self, cam, threshold=None):
        """
        获取激活的面部区域
        """



        threshold = np.percentile(cam, 75)
        active_regions = []
        region_scores = {}

        for region_name, (y1, y2, x1, x2) in self.regions.items():
            # 确保坐标在范围内
            y1, y2 = max(0, y1), min(cam.shape[0], y2)
            x1, x2 = max(0, x1), min(cam.shape[1], x2)

            if y1 < y2 and x1 < x2:
                region_score = cam[y1:y2, x1:x2].mean()
                region_scores[region_name] = region_score

                if region_score > threshold:
                    active_regions.append(region_name)

        return active_regions, region_scores

    def visualize_regions(self, image, cam, active_regions, save_path=None):
        """
        可视化热力图和激活区域
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 原始图像
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # 热力图
        axes[1].imshow(image)
        axes[1].imshow(cam, alpha=0.5, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')

        # 标注激活区域
        axes[2].imshow(image)
        for region in active_regions:
            y1, y2, x1, x2 = self.regions[region]
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 fill=False, edgecolor='red', linewidth=2)
            axes[2].add_patch(rect)
            axes[2].text(x1, y1 - 5, region.replace('_', ' '),
                         fontsize=8, color='red')
        axes[2].set_title(f'Active Regions ({len(active_regions)})')
        axes[2].axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


# -----------------------------
# 5. 训练函数
# -----------------------------
def train_model(model, train_loader, val_loader, epochs=EPOCHS):
    criterion = nn.MSELoss()  # 使用MSE损失
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
        for apex, onset in pbar:
            apex, onset = apex.to(device), onset.to(device)

            pred = model(apex)
            loss = criterion(pred, onset)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': train_loss / (pbar.n + 1)})

        avg_train_loss = train_loss / len(train_loader)

        # 验证
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for apex, onset in val_loader:
                apex, onset = apex.to(device), onset.to(device)
                pred = model(apex)
                loss = criterion(pred, onset)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch + 1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss
            }, os.path.join(SAVE_DIR, 'best_unet_model.pth'))
            print(f"✅ 保存最佳模型 (Val Loss: {avg_val_loss:.4f})")

    return model


# -----------------------------
# 6. 推理函数
# -----------------------------
def analyze_expression(model, image_path, threshold=0.5):
    """
    分析单张图像的面部激活区域
    """
    model.eval()

    # 加载图像
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_img = img.copy()

    # 预处理
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img_tensor = transform(img).unsqueeze(0).to(device)

    # 生成Grad-CAM
    cam_generator = GradCAM(model)
    cam = cam_generator.generate(img_tensor)

    # 分析区域
    analyzer = FaceRegionAnalyzer()
    active_regions, region_scores = analyzer.get_active_regions(cam, threshold)

    # 可视化
    analyzer.visualize_regions(original_img, cam, active_regions)

    # 生成文本描述
    region_names = [r.replace('_', ' ') for r in active_regions]
    if region_names:
        prompt = f"Active facial regions: {', '.join(region_names)}"
    else:
        prompt = "No significant active regions detected"

    print(f"\n📊 分析结果:")
    print(f"   {prompt}")
    print(f"\n📈 区域激活分数:")
    for region, score in sorted(region_scores.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"   {region}: {score:.3f}")

    return prompt, active_regions, region_scores


# -----------------------------
# 7. 主程序
# -----------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("面部表情区域定位系统 (U-Net + Grad-CAM)")
    print("=" * 60)

    # 数据预处理
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
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

    # 加载数据集
    try:
        full_dataset = ApexOnsetDataset(DATA_DIR, transform=train_transform, augment=True)

        # 划分训练集和验证集
        from sklearn.model_selection import train_test_split

        train_indices, val_indices = train_test_split(
            range(len(full_dataset)), test_size=0.2, random_state=42
        )

        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

        val_dataset.dataset.transform = val_transform

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        print(f"\n训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(val_dataset)}")

        # 创建模型
        model = ImprovedUNet().to(device)
        print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

        # 训练模型
        train_model(model, train_loader, val_loader, epochs=EPOCHS)

        # 测试推理
        print("\n" + "=" * 60)
        print("测试推理")
        print("=" * 60)

        # 找一个测试图像
        test_sample = full_dataset.samples[0][0]  # 第一个apex图像
        analyze_expression(model, test_sample)

    except Exception as e:
        print(f"错误: {e}")
        import traceback

        traceback.print_exc()