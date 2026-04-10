import json
import os

datadir = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/trainData1"
# input_file = 'z26TrainJsonl/casme2.jsonl'
# output_file = "z26trainData/RO/casmetrain.jsonl"
# os.makedirs(os.path.dirname(output_file), exist_ok=True)
# with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
#     for line in infile:
#         data = json.loads(line.strip())
#
#         # 构建新的图像路径
#         imgpatha = os.path.join(datadir, data['dataset'], data['subject'], data['filename'] + "/apex.jpg")
#         # data["imgpath"] = imgpath
#         imgpatho = os.path.join(datadir, data['dataset'], data['subject'], data['filename'] + "/onset.jpg")
#         # 检查路径是否存在
#         if not os.path.exists(imgpatha):
#             print(f"Warning: File does not exist - {imgpatha}")
#             continue
#         if not os.path.exists(imgpatho):
#             print(f"Warning: File does not exist - {imgpatho}")
#             continue
#         # 写入到输出文件
#         # print(imgpath)
#         new_data = {
#             "imgpatha": imgpatha,
#             "imgpatho": imgpatho,
#             "question": data.get("question", ""),
#             "answer": data.get("answer", "")
#         }
#         outfile.write(json.dumps(new_data, ensure_ascii=False) + "\n")
        # break




# input_file = 'z26TrainJsonl/samm.jsonl'
# output_file = "z26trainData/RO/sammtrain.jsonl"
# os.makedirs(os.path.dirname(output_file), exist_ok=True)
# with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
#     for line in infile:
#         data = json.loads(line.strip())
#
#         # 构建新的图像路径
#         subject_code = data['subject'].zfill(3)
#         filename = data['filename']
#         imagea_filename = "apex.jpg"
#         imgpatha = os.path.join(datadir, data['dataset'], subject_code, filename, imagea_filename)
#         imageo_filename = "onset.jpg"
#         imgpatho = os.path.join(datadir, data['dataset'], subject_code, filename, imageo_filename)
#         # data["imgpath"] = imgpath
#         # print(imgpatho)
#         # print(imgpatha)
#         # 检查路径是否存在
#         if not os.path.exists(imgpatha):
#             print(f"Warning: File does not exist - {imgpatha}")
#             continue
#         if not os.path.exists(imgpatho):
#             print(f"Warning: File does not exist - {imgpatho}")
#             continue
#         new_data = {
#             "imgpatha": imgpatha,
#             "imgpatho": imgpatho,
#             "question": data.get("question", ""),
#             "answer": data.get("answer", "")
#         }
#         # print(new_data)
#         # 写入到输出文件
#         outfile.write(json.dumps(new_data, ensure_ascii=False) + "\n")
#         # break




input_file = 'z26TrainJsonl/smic.jsonl'
output_file = "z26trainData/RO/smictrain.jsonl"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        data = json.loads(line.strip())

        # 构建新的图像路径
        if data['subject'][1] == "0":
            imgpath = os.path.join(datadir, data['dataset'], data['subject'][0]+data['subject'][2], data['filename'][0]+
                                   data['filename'][2:])
            # data["imgpath"] = imgpath
        else:
            imgpath = os.path.join(datadir, data['dataset'], data['subject'], data['filename'])
            # data["imgpath"] = imgpath
        # print(imgpath)
        imgpatha = os.path.join(imgpath+"/apex.jpg")
        imgpatho = os.path.join(imgpath + "/onset.jpg")
        # 检查路径是否存在
        if not os.path.exists(imgpatha):
            print(f"Warning: File does not exist - {imgpatha}")
            continue
        new_data = {
            "imgpatha": imgpatha,
            "imgpatho": imgpatho,
            "question": data.get("question", ""),
            "answer": data.get("answer", "")
        }
        # print(new_data)
        # 写入到输出文件
        outfile.write(json.dumps(new_data, ensure_ascii=False) + "\n")
