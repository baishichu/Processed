import json
import os

# 输入文件路径
input_file = '/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/z26TrainJsonl/1me_vqa_samm_casme2_smic2026.jsonl'

# 用于保存不同 dataset 的数据
output_data = {}

# 读取和处理
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line.strip())

        subject = data.get("subject", "")
        filename = data.get("filename", "")
        image_path = f"{subject}/{filename}"
        data["imagePath"] = image_path

        # 按 dataset 分类保存
        dataset = data.get("dataset", "unknown")
        if dataset not in output_data:
            output_data[dataset] = []
        output_data[dataset].append(data)

# 将每类 dataset 保存为独立 jsonl 文件
for dataset, items in output_data.items():
    output_file = f"z26TrainJsonl/{dataset}.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in items:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
# print("处理完成，生成文件：", ", ".join(f"{k}.jsonl" for k in output_data.keys()))
