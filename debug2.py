import json
import os
input_file = "trainData/train.jsonl"
output_file = "trainData/train1.jsonl"
with open(input_file, 'r', encoding='utf-8') as fin, \
     open(output_file, 'w', encoding='utf-8') as fout:
    for line in fin:
        data = json.loads(line)
        filtered = {
            "question": data.get("question"),
            "answer": data.get("answer"),
            "imgpath": data.get("imgpath")
        }
        fout.write(json.dumps(filtered, ensure_ascii=False) + '\n')