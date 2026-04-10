import cv2
import numpy as np
import json
import os

def compute_motion_intensity(apex, onset):
    apex = apex.astype(np.float32)
    onset = onset.astype(np.float32)

    diff = np.abs(onset - apex)

    diff_gray = np.mean(diff, axis=2)


    motion_intensity = float(diff_gray.mean())

    return motion_intensity, diff_gray



def compute_optical_flow_intensity(apex, onset):

    apex_gray = cv2.cvtColor(apex, cv2.COLOR_RGB2GRAY)
    onset_gray = cv2.cvtColor(onset, cv2.COLOR_RGB2GRAY)

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

    magnitude = np.sqrt(fx**2 + fy**2)

    flow_intensity = float(magnitude.mean())

    return flow_intensity, magnitude



def analyze_motion(apex_path, onset_path, img_size=224):

    apex = cv2.imread(apex_path)
    onset = cv2.imread(onset_path)

    if apex is None or onset is None:
        raise ValueError("图像读取失败")

    # BGR → RGB
    apex = cv2.cvtColor(apex, cv2.COLOR_BGR2RGB)
    onset = cv2.cvtColor(onset, cv2.COLOR_BGR2RGB)

    apex = cv2.resize(apex, (img_size, img_size))
    onset = cv2.resize(onset, (img_size, img_size))

    motion_intensity, diff_map = compute_motion_intensity(apex, onset)
    flow_intensity, flow_map = compute_optical_flow_intensity(apex, onset)


    print("\n==============================")
    print("📊 Motion Analysis Result:")
    print(f"Motion intensity: {motion_intensity:.4f}")
    print(f"Optical flow intensity: {flow_intensity:.4f}")


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



if __name__ == "__main__":
    # input_file = "regionssamm_test.jsonl"
    # output_file = "1regionssamm_test.jsonl"
    # base_dir = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/test1/testdata"
    input_file = "regionscasme_test.jsonl"
    output_file = "1regionscasme_test.jsonl"
    base_dir = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/test1/testdata"

    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            data = json.loads(line.strip())

            video = data["video"]
            video_id = data["video_id"]
            question = data["question"]
            answer = data["answer"]
            textAU = data["textAU"]


            image_dir1 = os.path.join(base_dir, video)
            onset_path = os.path.join(image_dir1, "onset.jpg")
            apex_path = os.path.join(image_dir1, "apex.jpg")
            print(onset_path)
            m = analyze_motion(apex_path, onset_path)

            re1 = analyze_motion(apex_path, onset_path)
            re2 = re1["answer"]
            # print("+++++++++++++++++++++", re1)
            re = {
                "video": video,
                "video_id": video_id,
                "question": question,
                "answer": answer,
                "textAU": textAU,
                "motionAndOptical": re2
            }
            print("=========================\n",  re)
            json.dump(re, fout, ensure_ascii=False)
            fout.write("\n")
            # break


