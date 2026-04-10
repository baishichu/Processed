import json
import os

input_file = "samm.jsonl"
output_file = "sammtrain.jsonl"

base_dir = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/trainData1"

with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
    for line in fin:
        data = json.loads(line.strip())

        dataset = data["dataset"]
        subject = data["subject"].zfill(3)  #
        filename = data["filename"]

        image_dir = os.path.join(base_dir, dataset, subject, filename)
        imagea = os.path.join(image_dir, "apex.jpg")
        imageo = os.path.join(image_dir, "onset.jpg")

        new_data = {
            "question": data["question"],
            "answer": data["answer"],
            "imagea": imagea,
            "imageo": imageo
        }

        json.dump(new_data, fout, ensure_ascii=False)
        fout.write("\n")
