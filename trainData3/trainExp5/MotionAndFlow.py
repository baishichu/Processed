import cv2
import numpy as np
import json
import os
# -----------------------------
# 输入：峰值帧和起始帧，输出：两帧之间的图像差分
# 构建答案：Motion intensity：3.24（特征的差值）
# 光流强度：平均光流的值 3.44
# =============================
# 1. 图像差分强度（Motion Intensity）
# -----------------------------
def compute_motion_intensity(apex, onset):
    apex = apex.astype(np.float32)
    onset = onset.astype(np.float32)

    diff = np.abs(onset - apex)

    # 转灰度
    diff_gray = np.mean(diff, axis=2)

    # 👉 强度 = 平均像素变化
    motion_intensity = float(diff_gray.mean())

    return motion_intensity, diff_gray


# -----------------------------
# 2. 光流强度（Optical Flow）
# -----------------------------
def compute_optical_flow_intensity(apex, onset):
    # 转灰度
    apex_gray = cv2.cvtColor(apex, cv2.COLOR_RGB2GRAY)
    onset_gray = cv2.cvtColor(onset, cv2.COLOR_RGB2GRAY)

    # Farneback 光流
    flow = cv2.calcOpticalFlowFarneback(
        apex_gray,
        onset_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    # flow: (H, W, 2)
    fx, fy = flow[..., 0], flow[..., 1]

    # 👉 光流幅值
    magnitude = np.sqrt(fx**2 + fy**2)

    # 👉 平均光流强度
    flow_intensity = float(magnitude.mean())

    return flow_intensity, magnitude


# -----------------------------
# 3. 主函数
# -----------------------------
def analyze_motion(apex_path, onset_path, img_size=224):
    # 读取图像
    apex = cv2.imread(apex_path)
    onset = cv2.imread(onset_path)

    if apex is None or onset is None:
        raise ValueError("图像读取失败")

    # BGR → RGB
    apex = cv2.cvtColor(apex, cv2.COLOR_BGR2RGB)
    onset = cv2.cvtColor(onset, cv2.COLOR_BGR2RGB)

    # resize（保证一致）
    apex = cv2.resize(apex, (img_size, img_size))
    onset = cv2.resize(onset, (img_size, img_size))

    # -----------------------------
    # 计算指标
    # -----------------------------
    motion_intensity, diff_map = compute_motion_intensity(apex, onset)
    flow_intensity, flow_map = compute_optical_flow_intensity(apex, onset)

    # -----------------------------
    # 输出结果
    # -----------------------------
    print("\n==============================")
    print("📊 Motion Analysis Result:")
    print(f"Motion intensity: {motion_intensity:.4f}")
    print(f"Optical flow intensity: {flow_intensity:.4f}")

    # -----------------------------
    # 构建答案（给Qwen用）
    # -----------------------------
    answer = (
        f"Motion intensity: {motion_intensity:.2f}. "
        f"Optical flow intensity: {flow_intensity:.2f}."
    )

    print("\n🧠 构建答案：")
    print(answer)

    return {
        "motion_intensity": motion_intensity,
        "flow_intensity": flow_intensity,
        "answer": answer,
        "diff_map": diff_map,
        "flow_map": flow_map
    }


# -----------------------------
# 4. 测试
# -----------------------------
if __name__ == "__main__":
    input_file = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/trainData3/trainExp5/regionssamm.jsonl"
    output_file = "motionsamm.jsonl"
    base_dir = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/trainData3/data3/SAMM"
    # input_file = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/trainData3/trainExp5/regionscasme.jsonl"
    # output_file = "motioncasme.jsonl"
    # base_dir = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/trainData3/data3/csame"

    # 打开输入文件和输出文件
    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            data = json.loads(line.strip())

            dataset = data["dataset"]
            subject = data["subject"].zfill(3)  # 统一补0如 "006"
            filename = data["filename"]
            question = data["question"]
            answer = data["answer"]
            textAU = data["textAU"]
            activate = data["activate"]


            image_dir1 = os.path.join(base_dir, subject, filename)
            onset_path = os.path.join(image_dir1, "onset.jpg")
            apex_path = os.path.join(image_dir1, "apex.jpg")
            print(onset_path)
            m = analyze_motion(apex_path, onset_path)

            re1 = analyze_motion(apex_path, onset_path)
            re2 = re1["answer"]
            # print("+++++++++++++++++++++", re1)
            re = {
                "dataset": dataset,
                "subject": subject,
                "filename": filename,
                "question": question,
                "answer": answer,
                "textAU": textAU,
                "activate": activate,
                "motionAndOptical": re2
            }
            # print("=========================\n",  re)
            json.dump(re, fout, ensure_ascii=False)
            fout.write("\n")
            # break


