# test_dataset.py
import os
import sys

# 添加路径
sys.path.insert(0, '/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/trainData3/trainExp5')

from trainEmo import MicroExpressionDataset, train_transform

# 测试数据加载
data_dir = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/trainData3/data3"

print("测试数据加载...")
try:
    dataset = MicroExpressionDataset(data_dir, transform=train_transform)
    print(f"\n✅ 成功加载 {len(dataset)} 个样本")

    # 测试第一个样本
    if len(dataset) > 0:
        img, label = dataset[0]
        print(f"第一个样本形状: {img.shape}")
        print(f"第一个样本标签: {label}")

except Exception as e:
    print(f"❌ 错误: {e}")