import json
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os
from PIL import Image


model = Qwen3VLForConditionalGeneration.from_pretrained(
    "/media/ph/29321BB58B527E5A/qwen/qwen3vl4",
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True  # Qwen3-VL 需要
)
model.eval()


processor = AutoProcessor.from_pretrained(
    "/media/ph/29321BB58B527E5A/qwen/qwen3vl4",
    trust_remote_code=True
)


picdir = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/test2/TV"
dir = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/MEGC2026_ME_VQA_Test/me_vqa_casme3_v2_test_to_answer.jsonl"
# dir = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/MEGC2026_ME_VQA_Test/me_vqa_samm_v2_test_to_answer.jsonl"

save = dir.split('/')[-1]
jsonl_name = save.split('_')[2]
save_name = jsonl_name + '_3_26.jsonl'
print(save_name)


data = []
with open(dir, 'r') as f:
    for line in f:
        item = json.loads(line.strip())
        video_name = item["video"]
        image_path = os.path.join(picdir, f"{video_name}.jpg")
        item['image'] = image_path
        data.append(item)

print(data[1])


result = []
for idx, item in enumerate(data):
    print(f"Processing {idx + 1}/{len(data)}")

    image_path = item["image"]

    image = Image.open(image_path).convert("RGB")


    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": f"The first image is the first frame of the video, and the second image is the optical flow image. Please answer the question: {item['question']}"
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    # 推理
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=None,
        )

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
        json_line = json.dumps(item, ensure_ascii=False)
        f.write(json_line + "\n")

print(f"save: {save_name}")