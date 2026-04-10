import json
import os

input_file = "smic.jsonl"
output_file = "smictrain3.jsonl"

base_dir1 = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/trainData1"
base_dir2 = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/trainData2"

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

        print(subject_path)
        image_dir1 = os.path.join(base_dir1, "smic", subject_path, f"{subject_path}{filename[3:]}")
        image1 = os.path.join(image_dir1, "onset.jpg")
        print(image1)
        image_dir2 = os.path.join(base_dir2, "smic", subject_path, f"{subject_path}{filename[3:]}")
        image2 = os.path.join(image_dir2, f"{subject_path}{filename[3:]}.jpg")
        print(image2)

        new_data = {
            "question": data["question"],
            "answer": data["answer"],
            "image1": image1,
            "image2": image2
        }

        json.dump(new_data, fout, ensure_ascii=False)
        fout.write('\n')
        # break
