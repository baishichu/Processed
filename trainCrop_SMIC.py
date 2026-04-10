# train 2
import re
from data_tools import Module_crop_times as mycrop
from data_tools import tool_helper as hp
import cv2
import json
import os
import pandas as pd
import numpy as np
import csv

def get_videos(dir):
    videos = {}
    for dirpath, _, filenames in os.walk(dir):
        pho_files = [f for f in filenames if f.endswith('.bmp')]
        if pho_files:
            frames = [os.path.join(dirpath, f) for f in pho_files]

            def extract_number(fpath):
                fname = os.path.basename(fpath)
                match = re.search(r'\d+', fname)
                return int(match.group()) if match else -1

            frames_sorted = sorted(frames, key=extract_number)
            videos[dirpath] = frames_sorted
    return videos


if __name__ == "__main__":
    crop_times = 2
    resize = 227
    dir = "/media/ph/208B-304E/train/SMIC/SMIC_all_raw/HS"  # SMIC
    videos = get_videos(dir)

    output_csv = "smic.csv"
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['ME Type'])

    for video, frames in videos.items():
        print(video)
        print(frames[0])
        onep = int(re.search(r"image(\d+)\.bmp$", frames[0]).group(1))
        # print(f"onep: {onep}########{frames[0][-10:-4]}")
        frame_0 = cv2.imread(frames[0])
        lll = len(frames)

        ymin, ymax, xmin, xmax = mycrop.crop_times(frame_0, crop_times)
        save_folder = '/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/trainData5/ps/smic/{}/{}'.format(
            video.split('/')[-4],video.split('/')[-1])
        if not os.path.exists(save_folder): os.makedirs(save_folder)
        print("*****", len(frames))
        # 查询条件
        subject = str((video.split('/')[-1]))
        df = pd.read_excel("trainData/SMIC.xlsx")
        df.columns = df.columns.str.strip()
        df['ME Type'] = df['ME Type'].astype(str)
        # 使用布尔索引查询 ApexFrame 和 OnsetFrame
        row = df[(df['ME Type'] == subject)]

        if not row.empty:

            offset = int(row['offset'].values[0])
            onset = int(row['onset'].values[0])
            print(f"ApexFrame: {offset}, OnsetFrame: {onset}")
            print(f"start: {onset-onset},end: {offset-onset}")
            subject_val = row['ME Type'].values[0]
            ee = subject_val.split('_')[1]
            if ee == "sur" or ee == "po":
                with open(output_csv, mode='a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow([subject_val])
        else:
            print("未找到对应的记录")
            continue
        #
        # print("phOnset", frames[1])
        # print("phApex", frames[-1])



        # break
        print(f"phOnset: {frames[onset-onep]},phApex: {frames[onset//3+offset*2//3-onep]}")
        # print(frames[-1])
        if ee == "sur" or ee == "po":
            phOnset = cv2.imread(frames[onset-onep+2])
            phApex = cv2.imread(frames[onset//3+offset*2//3-onep - 2])
        else:
            continue


        star1 = phOnset[ymin:ymax, xmin:xmax]
        end1 = phApex[ymin:ymax, xmin:xmax]

        star = cv2.resize(star1, (resize, resize))
        end = cv2.resize(end1, (resize, resize))

        prev_gray = cv2.cvtColor(star, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(end, cv2.COLOR_BGR2GRAY)

        # cv2.imwrite(os.path.join(save_folder, "onset.jpg"), prev_gray)
        # cv2.imwrite(os.path.join(save_folder, "apex.jpg"), next_gray)
        #
        # 使用特征点匹配估计全局运动
        orb = cv2.ORB_create(500)
        kp1, des1 = orb.detectAndCompute(prev_gray, None)
        kp2, des2 = orb.detectAndCompute(next_gray, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # 提取匹配点坐标
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # 估计仿射变换矩阵
        M, inliers = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.RANSAC, ransacReprojThreshold=5.0)

        # 将其对齐到前一帧
        h, w = prev_gray.shape
        next_warped = cv2.warpAffine(next_gray, M, (w, h))

        # 创建 TV-L1 光流估计器
        optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
        optical_flow.setTau(0.25)  # 时间步长 (time step)，较小值意味着收敛更稳定，但可能更慢。
        optical_flow.setLambda(0.2)  # TV-L1 模型的平滑项权重 (smoothness weight)，平衡光滑和数据项。
        optical_flow.setTheta(0.4)  # 主要用于增稳，取值通常在 0.3~0.5。
        optical_flow.setScalesNumber(5)  # 金字塔层数，层数多可更好处理大位移，但速度更慢。
        optical_flow.setWarpingsNumber(5)  # 每层金字塔的迭代数。
        optical_flow.setEpsilon(0.01)  # 迭代收敛的容差，越小收敛精度更高，代价是更慢。
        optical_flow.setInnerIterations(30)  # 每层光流迭代的内循环次数。
        optical_flow.setOuterIterations(10)  # 每层光流迭代的外循环次数。
        optical_flow.setScaleStep(0.8)  # 金字塔层之间的缩放系数。
        optical_flow.setGamma(0.0)  # 亮度项的平滑参数（对比度权重）。
        optical_flow.setMedianFiltering(5)  # 中值滤波核大小（0 关闭中值滤波，>0 启用）。
        optical_flow.setUseInitialFlow(False)  # 是否使用初始光流（可选，若你有初始光流预测）。

        # 计算光流
        flow = optical_flow.calc(prev_gray, next_warped, None)
        # 计算平均光流
        mean_flow = np.mean(flow.reshape(-1, 2), axis=0)
        # 减去平均光流
        flow = flow - mean_flow
        # # 双边滤波
        # flow[..., 0] = cv2.bilateralFilter(flow[..., 0], d=9, sigmaColor=75, sigmaSpace=75)
        # flow[..., 1] = cv2.bilateralFilter(flow[..., 1], d=9, sigmaColor=75, sigmaSpace=75)
        # 高斯滤波
        flow[..., 0] = cv2.GaussianBlur(flow[..., 0], (5, 5), 0)
        flow[..., 1] = cv2.GaussianBlur(flow[..., 1], (5, 5), 0)

        # 可视化光流
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        hsv = np.zeros((prev_gray.shape[0], prev_gray.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # 与灰度图映射
        edges = cv2.Canny(prev_gray, 40, 120)
        edges_dilated = cv2.dilate(edges, None, iterations=2)
        flow_rgb_masked = cv2.bitwise_and(flow_rgb, flow_rgb, mask=edges_dilated)
        # cv2.imwrite("TV-L1V4-flow.png", flow_rgb_masked)

        # 灰度图转三通道rgb
        prev_gray_rgb = cv2.cvtColor(next_gray, cv2.COLOR_GRAY2BGR)
        # 按一定比例混合光流可视化结果和原灰度图
        alpha = 0.7  # 0～1
        overlay = cv2.addWeighted(prev_gray_rgb, alpha, flow_rgb_masked, 1 - alpha, 0)
        img_name = "{}".format(video.split('/')[-1])
        save_path = '{}/{}.jpg'.format(save_folder, img_name)
        print(save_path)
        # 保存光流图像
        # cv2.imwrite(save_path, flow_rgb_masked)
        # 保存叠加图像
        cv2.imwrite(save_path, overlay)

        # break
    print('Finish all.')
