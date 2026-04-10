import json
import os

input_file = "smictrain3_valid.jsonl"

missing_count = 0

with open(input_file, 'r', encoding='utf-8') as fin:
    for line in fin:
        data = json.loads(line.strip())

        image = data.get("image1", "")

        missing = []

        if not os.path.exists(image):
            missing.append(f"Missing imagea: {image}")

        if missing:
            missing_count += 1
            print(f"[Missing #{missing_count}] ID: {data.get('id', 'N/A')}")
            for m in missing:
                print("  ", m)
            print("-" * 50)

print(f"done：{missing_count}")
