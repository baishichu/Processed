# test three get Result
import json
from pyexpat.errors import messages
from peft import PeftModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os
from PIL import Image

# 1. 加载 base Qwen2.5-VL 模型
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/media/ph/208B-304E/llamaFactory/LLaMA-Factory/Model/Qwen2.5-VL-7B-Instruct",
    dtype=torch.bfloat16,
    device_map="auto",
)

# 2. 加载 LoRA adapter（路径为 LLaMA-Factory 的保存目录）
lora_path = "/media/ph/208B-304E/llamaFactory/LLaMA-Factory/examples/train_lora/saves/ROandP/qwen2_5vl-7b/lora/sft/checkpoint-472"
model = PeftModel.from_pretrained(base_model, lora_path)
model.eval()
# default processer
processor = AutoProcessor.from_pretrained(
    "/media/ph/208B-304E/llamaFactory/LLaMA-Factory/Model/Qwen2.5-VL-7B-Instruct")

# Messages containing multiple images and a text query
picdir = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/test3/testdata"
# dir = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/MEGC2026_ME_VQA_Test/me_vqa_casme3_v2_test_to_answer.jsonl"
dir = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/MEGC2026_ME_VQA_Test/me_vqa_samm_v2_test_to_answer.jsonl"
save = dir.split('/')[-1]
jsonl_name = save.split('_')[2]
save_name = jsonl_name + 'q257.jsonl'
print(save_name)
data = []

with open(os.path.join(dir), 'r') as f:
    for line in f:
        item = json.loads(line.strip())
        video_name = item["video"]
        image_path = os.path.join(picdir, f"{video_name}.jpg")
        item['image'] = image_path
        data.append(item)
print(data[1])
result = []

for item in data:
    # 构造 prompt
    prompt = f"The first image is the first frame of the video, and the second image is the optical flow image. <image><image>Please answer the question:{item['question']}"
    image_path = item["image"]
    # 加载灰度图像为 PIL.Image 格式
    image = Image.open(image_path).convert("RGB")

    # 构造文本模板，使用两个图像占位符
    text = processor.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": item['question']}
        ]}],
        tokenize=False,
        add_generation_prompt=True
    )

    # 处理输入（1张图 + 文本）
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    # 推理
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    item["answer"] = output_text[0]

    del item["image"]

    print(json.dumps(item, ensure_ascii=False))
    result.append(item)
    # break

with open(save_name, 'w', encoding="utf-8") as f:
    for item in result:
        json_line = json.dumps(item, ensure_ascii=False)
        f.write(json_line + "\n")
