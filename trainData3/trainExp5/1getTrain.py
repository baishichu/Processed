import json
import os

# input_file = 'motioncasme.jsonl'

# output_file = '1casme2train3.jsonl'

input_file = 'motionsamm.jsonl'

output_file = '1sammtrain3.jsonl'


base_dir = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/trainData3/data3/SAMM"

with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
    for line in fin:
        data = json.loads(line.strip())

        subject = data.get("subject", "")
        filename = data.get("filename", "")
        dataset = data.get("dataset", "")
        sub = data.get("subject", "")
        base_path1 = os.path.join(base_dir, subject, filename)
        imgname = "{}_{}".format(sub,filename)
        image1 = os.path.join(base_path1, f"{imgname}pr.jpg")
        print(image1)
        if os.path.exists(image1):
            print("文件存在")
        else:
            print("文件不存在")
        # 提取需要的字段
        output_data = {
            "question": data.get("question", ""),
            "answer": data.get("answer", ""),
            "image1": image1,
            "textAU": data["textAU"],
            "motionAndOptical": data["motionAndOptical"]
        }
        print(output_data)

        json.dump(output_data, fout, ensure_ascii=False)
        fout.write('\n')
        # break

print(f"SaveAs：{output_file}")
