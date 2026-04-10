# test three get Result
import json
from pyexpat.errors import messages
from peft import PeftModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os
from PIL import Image

base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/media/ph/29321BB58B527E5A/qwen/qwen25vl3",
    dtype=torch.bfloat16,
    device_map="auto",
)

lora_path = "/media/ph/208B-304E/llamaFactory/LLaMA-Factory/saves/ROW/qwen2_5vl-3b/PPRlora/sft"
model = PeftModel.from_pretrained(base_model, lora_path)
model.eval()
# default processer
processor = AutoProcessor.from_pretrained(
    "/media/ph/29321BB58B527E5A/qwen/qwen25vl3")

# Messages containing multiple images and a text query
picdir = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/test3/testdata"
# dir = "1regionscasme_test.jsonl"
dir = "1regionssamm_test.jsonl"
save_name = 'sammq2535.jsonl'
# save_name = 'casmeq2535.jsonl'
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
    image_path = item["image"]

    image = Image.open(image_path).convert("RGB")

    text = processor.apply_chat_template(
        [{
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        "Facial movement description:\n"
                        f"{item['textAU']}.\n\n"
                        "Motion information:\n"
                        f"{item['motionAndOptical']}\n\n"
                        f"Question: {item['question']}"
                    )
                }
            ]
        }],
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=64)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    item["answer"] = output_text[0]

    del item["image"]

    print(json.dumps(item, ensure_ascii=False))
    result.append(item)
with open(save_name, 'w', encoding="utf-8") as f:
    for item in result:
        video = item["video"]
        video_id = item["video_id"]
        question = item["question"]
        answer = item["answer"]
        new_data = {
            "video": video,
            "video_id": video_id,
            "question": question,
            "answer": answer
        }
        json_line = json.dumps(new_data, ensure_ascii=False)
        f.write(json_line + "\n")
