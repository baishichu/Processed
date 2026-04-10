import json
import os


input_file = 'casme2.jsonl'

output_file = 'casme2train.jsonl'


base_dir = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/trainData1"

with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
    for line in fin:
        data = json.loads(line.strip())


        subject = data.get("subject", "")
        filename = data.get("filename", "")
        dataset = data.get("dataset", "")
        base_path = os.path.join(base_dir, dataset, subject, filename)
        imagea = os.path.join(base_path, "apex.jpg")
        imageo = os.path.join(base_path, "onset.jpg")


        output_data = {
            "question": data.get("question", ""),
            "answer": data.get("answer", ""),
            "imagea": imagea,
            "imageo": imageo
        }


        json.dump(output_data, fout, ensure_ascii=False)
        fout.write('\n')

print(f"saveAs：{output_file}")
