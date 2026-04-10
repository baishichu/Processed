import json
import os

input_file = "smic.jsonl"
output_file = "smictrain.jsonl"

base_dir = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/trainData1"

with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
    for line in fin:
        data = json.loads(line.strip())

        if data["dataset"] != "smic":
            continue

        subject = data["subject"]
        filename = data["filename"]

        if subject.startswith("s0"):
            subject_path = "s" + str(int(subject[1:]))  # s01 -> s1
        else:
            subject_path = subject

        image_dir = os.path.join(base_dir, "smic", subject_path, f"{subject_path}{filename[3:]}")
        imagea = os.path.join(image_dir, "apex.jpg")
        imageo = os.path.join(image_dir, "onset.jpg")

        new_data = {
            "question": data["question"],
            "answer": data["answer"],
            "imagea": imagea,
            "imageo": imageo
        }

        json.dump(new_data, fout, ensure_ascii=False)
        fout.write('\n')
