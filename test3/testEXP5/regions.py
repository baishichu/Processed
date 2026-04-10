import os
import cv2
import numpy as np
import json

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


def get_motion_map(apex, onset):
    apex = apex.astype(np.float32)
    onset = onset.astype(np.float32)

    diff = np.abs(onset - apex)
    gray = np.mean(diff, axis=2)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    gray = gray / (gray.max() + 1e-8)

    return gray



def get_region_motion(apex, onset, regions):
    motion_map = get_motion_map(apex, onset)

    region_info = {}

    for name, (y1, y2, x1, x2) in regions.items():
        region = motion_map[y1:y2, x1:x2]

        if region.size == 0:
            continue

        intensity = float(region.mean())

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


def intensity_to_word(score):
    if score > 0.4:
        return "strongly"
    elif score > 0.25:
        return "moderately"
    elif score > 0.1:
        return "slightly"
    else:
        return None



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


def analyze_expression(apex_path, onset_path, img_size=224):
    analyzer = FaceRegionAnalyzer(img_size)

    apex = cv2.imread(apex_path)
    onset = cv2.imread(onset_path)

    if apex is None or onset is None:
        raise ValueError("图像读取失败")


    apex = cv2.cvtColor(apex, cv2.COLOR_BGR2RGB)
    onset = cv2.cvtColor(onset, cv2.COLOR_BGR2RGB)

    apex = cv2.resize(apex, (img_size, img_size))
    onset = cv2.resize(onset, (img_size, img_size))

    region_info = get_region_motion(apex, onset, analyzer.regions)


    description = build_natural_description(region_info)

    print("\n==============================")
    print("🧠 自然语言描述：")
    print(description)

    print("\n📊 Top区域强度：")
    for k, v in sorted(region_info.items(), key=lambda x: x[1]["intensity"], reverse=True)[:5]:
        print(f"{k}: {v['intensity']:.3f}")

    return description, region_info



if __name__ == "__main__":
    # dir = ""
    dir = ""
    input_file = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/MEGC2026_ME_VQA_Test/me_vqa_samm_v2_test_to_answer.jsonl"
    output_file = "regionssamm_test.jsonl"
    base_dir = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/test1/testdata"
    # input_file = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/MEGC2026_ME_VQA_Test/me_vqa_casme3_v2_test_to_answer.jsonl"
    # output_file = "regionscasme_test.jsonl"
    # base_dir = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/test1/testdata"



    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            data = json.loads(line.strip())
            video = data["video"]
            question = data["question"]
            answer = data["answer"]
            video_id = data["video_id"]

            image_dir1 = os.path.join(base_dir, video)
            onset_path = os.path.join(image_dir1, "onset.jpg")
            apex_path = os.path.join(image_dir1, "apex.jpg")
            print(onset_path)

            re1, re2 = analyze_expression(apex_path, onset_path)
            print("+++++++++++++++++++++\n", re1)
            re = {
                "video_id": video_id,
                "video": video,
                "question": question,
                "answer": answer,
                "textAU": re1
            }
            json.dump(re, fout, ensure_ascii=False)
            fout.write("\n")
            # break
