import os
from PIL import Image

onset_root = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/test1/testdata"
flow_root = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/test2/TV"
save_root = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/test3/testdata"

os.makedirs(save_root, exist_ok=True)

for folder_name in os.listdir(onset_root):
    folder_path = os.path.join(onset_root, folder_name)
    if not os.path.isdir(folder_path):
        continue

    onset_path = os.path.join(folder_path, "onset.jpg")
    flow_path = os.path.join(flow_root, f"{folder_name}.jpg")

    if not os.path.exists(onset_path) or not os.path.exists(flow_path):
        print(f" {folder_name}, noFind")
        continue

    onset = Image.open(onset_path).convert("L")
    flow = Image.open(flow_path).convert("RGB")


    if onset.size != flow.size:
        onset = onset.resize(flow.size)


    r, g, _ = flow.split()
    b = onset


    fused = Image.merge("RGB", (r, g, b))


    save_path = os.path.join(save_root, f"{folder_name}_fused.jpg")
    fused.save(save_path)
    print(f"{folder_name} done -> {save_path}")
