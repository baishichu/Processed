import json
import os
# 构造测试集*****************************************************************************
# 假设原始测试集文件为 test.jsonl（每行一个 JSON）
input_file = '/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/MEGC2025_ME_VQA_Test/me_vqa_samm_test_to_answer.jsonl'
output_file = 'samm_test.json'

# 假设你的图片路径规则是：video_id 映射到图片路径
# 举例：CAS-1 -> /media/ph/208B-304E/qwen-vl-finetune/testData/CAS-1.jpg
image_base_path = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/TV"

formatted_data = []

with open(input_file, 'r') as f:
    for line in f:
        item = json.loads(line.strip())
        video_name = item['video']
        question = item['question']

        formatted_item = {
            "messages": [
                {"role": "user", "content": f"<image>{question}"},
                {"role": "assistant", "content": ""}
            ],
            "images": [os.path.join(image_base_path, f"{video_name}.jpg")]
        }

        formatted_data.append(formatted_item)

with open(output_file, 'w') as f:
    json.dump(formatted_data, f, indent=2)

print(f"转换完成，共 {len(formatted_data)} 条数据，保存至 {output_file}")
