import json
import os

input_file = "smictrain.jsonl"

missing_count = 0

with open(input_file, 'r', encoding='utf-8') as fin:
    for line in fin:
        data = json.loads(line.strip())

        imagea = data.get("imagea", "")
        imageo = data.get("imageo", "")

        missing = []

        if not os.path.exists(imagea):
            missing.append(f"Missing imagea: {imagea}")
        if not os.path.exists(imageo):
            missing.append(f"Missing imageo: {imageo}")

        if missing:
            missing_count += 1
            print(f"[Missing #{missing_count}] ID: {data.get('id', 'N/A')}")
            for m in missing:
                print("  ", m)
            print("-" * 50)

print(f"done：{missing_count}")

