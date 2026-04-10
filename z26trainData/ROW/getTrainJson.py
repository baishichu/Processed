import json
import os

datadir = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/trainData2"
# input_file = 'z26TrainJsonl/casme2.jsonl'
# output_file = "z26trainData/ROW/casmetrain.jsonl"
# os.makedirs(os.path.dirname(output_file), exist_ok=True)
# with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
#     for line in infile:
#         data = json.loads(line.strip())
#
#         # 构建新的图像路径
#         imgpath = os.path.join(datadir, data['dataset'], data['subject'], data['filename'], data['image_id'] + ".jpg")
#         # data["imgpath"] = imgpath
#
#         # 检查路径是否存在
#         if not os.path.exists(imgpath):
#             print(f"Warning: File does not exist - {imgpath}")
#             continue
#         # 写入到输出文件
#         # print(imgpath)
#         new_data = {
#             "imgpath": imgpath,
#             "question": data.get("question", ""),
#             "answer": data.get("answer", "")
#         }
#         outfile.write(json.dumps(new_data, ensure_ascii=False) + "\n")
#         # break




# input_file = 'z26TrainJsonl/samm.jsonl'
# output_file = "z26trainData/ROW/sammtrain.jsonl"
# os.makedirs(os.path.dirname(output_file), exist_ok=True)
# with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
#     for line in infile:
#         data = json.loads(line.strip())
#
#         # 构建新的图像路径
#         subject_code = data['subject'].zfill(3)
#         filename = data['filename']
#         image_filename = f"{subject_code}_{data['image_id']}.jpg"
#         imgpath = os.path.join(datadir, data['dataset'], subject_code, filename, image_filename)
#
#         # data["imgpath"] = imgpath
#
#         # 检查路径是否存在
#         if not os.path.exists(imgpath):
#             print(f"Warning: File does not exist - {imgpath}")
#             continue
#         # print(imgpath)
#         new_data = {
#             "imgpath": imgpath,
#             "question": data.get("question", ""),
#             "answer": data.get("answer", "")
#         }
#         # print(new_data)
#         # 写入到输出文件
#         outfile.write(json.dumps(new_data, ensure_ascii=False) + "\n")
#         # break




input_file = 'z26TrainJsonl/smic.jsonl'
output_file = "z26trainData/ROW/smictrain.jsonl"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        data = json.loads(line.strip())

        # 构建新的图像路径
        if data['subject'][1] == "0":
            imgpath = os.path.join(datadir, data['dataset'], data['subject'][0]+data['subject'][2], data['filename'][0]+
                                   data['filename'][2:], data['image_id'][0]+data['image_id'][2:] + ".jpg")
            # data["imgpath"] = imgpath
        else:
            imgpath = os.path.join(datadir, data['dataset'], data['subject'], data['filename'],data['image_id'] + ".jpg")
            # data["imgpath"] = imgpath
        # 检查路径是否存在
        if not os.path.exists(imgpath):
            print(f"Warning: File does not exist - {imgpath}")
            continue
        new_data = {
            "imgpath": imgpath,
            "question": data.get("question", ""),
            "answer": data.get("answer", "")
        }
        # print(new_data)
        # 写入到输出文件
        outfile.write(json.dumps(new_data, ensure_ascii=False) + "\n")
