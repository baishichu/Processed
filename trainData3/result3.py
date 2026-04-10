# test three get Result
import json
from pyexpat.errors import messages
from peft import PeftModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os
from PIL import Image

base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/Model/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

lora_path = "/media/ph/208B-304E/llamaFactory/LLaMA-Factory/saves/qwen2_5vl-7b1/lora/sft"
model = PeftModel.from_pretrained(base_model, lora_path)
model.eval()
# default processer
processor = AutoProcessor.from_pretrained(
    "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/Model/Qwen2.5-VL-7B-Instruct")

# Messages containing multiple images and a text query
picdir = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/test1/testdata"
picdir1 = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/test2/TV"
dir = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/MEGC2025_ME_VQA_Test/me_vqa_casme3_test_to_answer.jsonl"
# dir = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/MEGC2025_ME_VQA_Test/me_vqa_samm_test_to_answer.jsonl"
save = dir.split('/')[-1]
jsonl_name = save.split('_')[2]
save_name = jsonl_name + '.jsonl'
print(save_name)
data = []

with open(os.path.join(dir), 'r') as f:
    for line in f:
        item = json.loads(line.strip())
        video_name = item["video"]
        image_path1 = os.path.join(picdir, video_name, "onset.jpg")
        item['image1'] = image_path1
        image_path2 = os.path.join(picdir1, f"{video_name}.jpg")
        item['image2'] = image_path2
        data.append(item)
print(data[1])
result = []

for item in data:

    prompt = f"The first image is the onset frame of the micro-expression, and the second image is the apex frame. <image><image>Please answer the question:{item['question']}"

    image1 = Image.open(item["image1"]).convert("L")  # onset
    image_path=item["image1"]
    image2 = Image.open(image_path).convert("RGB")  # apex

    text = processor.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "image"},
            {"type": "image"},
            {"type": "text", "text": item['question']}
        ]}],
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[[image1, image2]],
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    item["answer"] = output_text[0]

    del item["image1"]
    del item["image2"]

    print(json.dumps(item, ensure_ascii=False))
    result.append(item)
    # break

with open(save_name, 'w', encoding="utf-8") as f:
    for item in result:
        json_line = json.dumps(item, ensure_ascii=False)
        f.write(json_line + "\n")
