import os
import cv2
import numpy as np
import json
### 这里获取了一个apex与onset之间的兴趣区域变化幅度
# -----------------------------
# 1. 面部区域定义（10区域）
# -----------------------------
class FaceRegionAnalyzer:
    def __init__(self, img_size=224):
        self.img_size = img_size

        # (y1, y2, x1, x2)
        self.regions = {
            'left_brow': (0, 80, 0, 112),
            'right_brow': (0, 80, 112, 224),
            'left_eye': (80, 130, 0, 112),
            'right_eye': (80, 130, 112, 224),
            'left_nose': (130, 170, 70, 112),
            'right_nose': (130, 170, 112, 154),
            'left_mouth_corner': (170, 200, 50, 100),
            'right_mouth_corner': (170, 200, 124, 174),
            'left_chin': (200, 224, 50, 100),
            'right_chin': (200, 224, 124, 174),
        }


# -----------------------------
# 2. 计算运动图（核心）
# -----------------------------
def get_motion_map(apex, onset):
    apex = apex.astype(np.float32)
    onset = onset.astype(np.float32)

    diff = np.abs(onset - apex)
    gray = np.mean(diff, axis=2)

    # 平滑
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 归一化
    gray = gray / (gray.max() + 1e-8)

    return gray


# -----------------------------
# 3. 区域运动分析（强度 + 方向）
# -----------------------------
def get_region_motion(apex, onset, regions):
    motion_map = get_motion_map(apex, onset)

    region_info = {}

    for name, (y1, y2, x1, x2) in regions.items():
        region = motion_map[y1:y2, x1:x2]

        if region.size == 0:
            continue

        intensity = float(region.mean())

        # 简单方向估计
        h, w = region.shape

        left_half = region[:, :w // 2].mean()
        right_half = region[:, w // 2:].mean()

        direction = "neutral"
        if left_half > right_half * 1.1:
            direction = "left"
        elif right_half > left_half * 1.1:
            direction = "right"

        region_info[name] = {
            "intensity": intensity,
            "direction": direction
        }

    return region_info


# -----------------------------
# 4. 强度 → 描述词
# -----------------------------
def intensity_to_word(score):
    if score > 0.4:
        return "strongly"
    elif score > 0.25:
        return "moderately"
    elif score > 0.1:
        return "slightly"
    else:
        return None


# -----------------------------
# 5. 区域 → 动作描述（核心）
# -----------------------------
def region_to_phrase(region, intensity, direction):
    level = intensity_to_word(intensity)
    if level is None:
        return None

    region_text = region.replace("_", " ")

    if "eye" in region:
        return f"{region_text} {level} closed"

    elif "brow" in region:
        return f"{region_text} {level} raised"

    elif "mouth corner" in region:
        if direction == "left":
            return f"{region_text} {level} pulled left"
        elif direction == "right":
            return f"{region_text} {level} pulled right"
        else:
            return f"{region_text} {level} raised"

    elif "nose" in region:
        return f"{region_text} {level} wrinkled"

    elif "chin" in region:
        return f"{region_text} {level} tensed"

    return f"{region_text} {level} moved"


# -----------------------------
# 6. 生成自然语言描述
# -----------------------------
def build_natural_description(region_info, top_k=3):
    sorted_regions = sorted(
        region_info.items(),
        key=lambda x: x[1]["intensity"],
        reverse=True
    )

    descriptions = []

    for region, info in sorted_regions[:top_k]:
        phrase = region_to_phrase(
            region,
            info["intensity"],
            info["direction"]
        )

        if phrase:
            descriptions.append(phrase)

    if not descriptions:
        return "No significant facial movement detected"

    return ", ".join(descriptions)


# -----------------------------
# 7. 主函数
# -----------------------------
def analyze_expression(apex_path, onset_path, img_size=224):
    analyzer = FaceRegionAnalyzer(img_size)

    # 读取图像
    apex = cv2.imread(apex_path)
    onset = cv2.imread(onset_path)

    if apex is None or onset is None:
        raise ValueError("图像读取失败")

    # 转RGB
    apex = cv2.cvtColor(apex, cv2.COLOR_BGR2RGB)
    onset = cv2.cvtColor(onset, cv2.COLOR_BGR2RGB)

    # resize（必须一致）
    apex = cv2.resize(apex, (img_size, img_size))
    onset = cv2.resize(onset, (img_size, img_size))

    # 获取区域信息
    region_info = get_region_motion(apex, onset, analyzer.regions)

    # 生成描述
    description = build_natural_description(region_info)

    print("\n==============================")
    print("🧠 自然语言描述：")
    print(description)

    print("\n📊 Top区域强度：")
    for k, v in sorted(region_info.items(), key=lambda x: x[1]["intensity"], reverse=True)[:5]:
        print(f"{k}: {v['intensity']:.3f}")

    return description, region_info


# -----------------------------
# 8. 测试入口
# -----------------------------
if __name__ == "__main__":
    # input_file = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/trainData2/samm.jsonl"
    # output_file = "regionssamm.jsonl"
    # base_dir = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/trainData3/data3/SAMM"
    input_file = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/trainData2/casme2.jsonl"
    output_file = "regionscasme.jsonl"
    base_dir = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/trainData3/data3/csame"


    # 打开输入文件和输出文件
    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            data = json.loads(line.strip())

            dataset = data["dataset"]
            subject = data["subject"].zfill(3)  # 统一补0如 "006"
            filename = data["filename"]
            question = data["question"]
            answer = data["answer"]

            image_dir1 = os.path.join(base_dir, subject, filename)
            onset_path = os.path.join(image_dir1, "onset.jpg")
            apex_path = os.path.join(image_dir1, "apex.jpg")
            print(onset_path)

            re1, re2 = analyze_expression(apex_path, onset_path)
            print("+++++++++++++++++++++", re1, re2)
            re = {
                "dataset": dataset,
                "subject": subject,
                "filename": filename,
                "question": question,
                "answer": answer,
                "textAU": re1,
                "activate": re2
            }
            json.dump(re, fout, ensure_ascii=False)
            fout.write("\n")
            # break
