# test two Flow picture
import cv2
import numpy as np
import json
import os


def get_videos(dir):
    videos = {}
    for dirpath, _, filenames in os.walk(dir):
        pho_files = [f for f in filenames if f.endswith('.jpg')]
        if pho_files:
            frames = [os.path.join(dirpath, f) for f in pho_files]
            frames_sorted = sorted(frames, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            videos[dirpath] = frames_sorted
    return videos


if __name__ == "__main__":

    crop_times = 2
    resize = 227
    file = "testdata"
    dir1 = os.path.join("/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/testData")
    print(dir1)
    dir = "/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/test1/testdata"
    videos = get_videos(dir1)
    for video, frames in videos.items():
        print(video, len(frames))
        save = '{}/{}'.format(dir, video.split('/')[-1])
        if not os.path.exists(save): os.makedirs(save)
        print("save", save)
        start = frames[0]
        # end = frames[-1]
        if len(frames) < 15:
            end = frames[int(len(frames) * 4 / 5)]
        else:
            end = frames[-1]
        print("start", start, "end", end)
        prev_gray = cv2.imread(start, cv2.IMREAD_GRAYSCALE)
        next_gray = cv2.imread(end, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(os.path.join(save, "onset.jpg"), prev_gray)
        cv2.imwrite(os.path.join(save, "apex.jpg"), next_gray)
        # break
