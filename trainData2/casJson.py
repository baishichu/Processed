import json
import os

input_file = 'casme2.jsonl'

output_file = 'casme2train3.jsonl'

base_dir1 = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/trainData1"
base_dir2 = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/trainData2"

with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
    for line in fin:
        data = json.loads(line.strip())

        subject = data.get("subject", "")
        filename = data.get("filename", "")
        dataset = data.get("dataset", "")

        base_path2 = os.path.join(base_dir2, dataset, subject, filename)
        base_path1 = os.path.join(base_dir1, dataset, subject, filename)
        image1 = os.path.join(base_path1, "onset.jpg")
        print(image1)
        image2 = os.path.join(base_path2, f"{subject}_{filename}.jpg")
        print(image2)

        output_data = {
            "question": data.get("question", ""),
            "answer": data.get("answer", ""),
            "image1": image1,
            "image2": image2
        }

        json.dump(output_data, fout, ensure_ascii=False)
        fout.write('\n')
        # break

print(f"saveAs：{output_file}")
