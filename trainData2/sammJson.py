import json
import os

input_file = "samm.jsonl"
output_file = "sammtrain3.jsonl"

base_dir1 = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/trainData1"
base_dir2 = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/trainData2"

with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
    for line in fin:
        data = json.loads(line.strip())

        dataset = data["dataset"]
        subject = data["subject"].zfill(3)
        filename = data["filename"]

        image_dir1 = os.path.join(base_dir1, dataset, subject, filename)
        image1 = os.path.join(image_dir1, "onset.jpg")
        print(image1)
        image_dir2 = os.path.join(base_dir2, dataset, subject, filename)
        image2 = os.path.join(image_dir2, f"{subject}_{filename}.jpg")
        print(image2)

        new_data = {
            "question": data["question"],
            "answer": data["answer"],
            "image1": image1,
            "image2": image2
        }
        # break

        json.dump(new_data, fout, ensure_ascii=False)
        fout.write("\n")
        # break
