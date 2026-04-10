import os
from PIL import Image
import cv2
import numpy as np

# 文件夹路径
onset_root = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/test1/testdata"
flow_root = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/test2/TV"
save_root = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/test3/testdata"

os.makedirs(save_root, exist_ok=True)

# 遍历 CAS-* / SAMM-* 文件夹
for folder_name in os.listdir(onset_root):
    folder_path = os.path.join(onset_root, folder_name)
    if not os.path.isdir(folder_path):
        continue

    onset_path = os.path.join(folder_path, "onset.jpg")
    flow_path = os.path.join(flow_root, f"{folder_name}.jpg")

    # 检查文件是否存在
    if not os.path.exists(onset_path) or not os.path.exists(flow_path):
        print(f"跳过 {folder_name}, 文件缺失")
        continue

    # 读取图像
    onset = cv2.imread(onset_path, cv2.IMREAD_GRAYSCALE)  # 灰度图
    flow = cv2.imread(flow_path, cv2.IMREAD_COLOR)        # 彩色光流图 BGR

    # 调整大小一致
    if onset.shape[:2] != flow.shape[:2]:
        onset = cv2.resize(onset, (flow.shape[1], flow.shape[0]))

    # 灰度图转成三通道
    onset_rgb = cv2.cvtColor(onset, cv2.COLOR_GRAY2BGR)

    # 按一定比例融合
    alpha = 0.7  # 灰度图权重
    beta = 1 - alpha  # 光流图权重
    fused = cv2.addWeighted(onset_rgb, alpha, flow, beta, 0)

    # 保存
    save_path = os.path.join(save_root, f"{folder_name}.jpg")
    cv2.imwrite(save_path, fused)
    print(f"{folder_name} 已处理 -> {save_path}")

print("批量合成完成！")