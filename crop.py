# test one crop picture

import os
import json
import cv2
from data_tools import Module_crop_times as mycrop
from data_tools import tool_helper as hp


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

    current_dir = os.path.dirname(os.path.realpath(__file__))
    dir = os.path.join(current_dir, 'raw_Data')
    videos = get_videos(dir)
    # print(json.dumps(videos, indent=4,ensure_ascii=False))
    for video, frames in videos.items():
        # print(video[-6:])
        # print(json.dumps(frames, indent=4,ensure_ascii=False))
        num_frames = len(frames)
        frame_0 = cv2.imread(frames[0])
        print(frames[0][-8:-4])
        ymin, ymax, xmin, xmax = mycrop.crop_times(frame_0, crop_times)
        save_folder = '/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/testData/{}'.format(video[-6:])
        if not os.path.exists(save_folder): os.makedirs(save_folder)
        save_folder_examples = '/media/ph/208B-304E/projectVQA/MEGC2025VQA/Processed/por_people/{}'.format(video[-6:])
        if not os.path.exists(save_folder_examples): os.makedirs(save_folder_examples)
        for i in range(num_frames):
            frame = cv2.imread(frames[i])
            crop_frame = frame[ymin:ymax, xmin:xmax]
            save_frame = cv2.resize(crop_frame, (resize, resize))

            if i == 0: cv2.imwrite('{}/{}.jpg'.format(save_folder_examples, video[-6:], frames[-8:-4]), save_frame)
            img_name = frames[i].split('/')[-1]
            save_path = '{}/{}'.format(save_folder, img_name)
            cv2.imwrite(save_path, save_frame)

            if (i + 1) % 1000 == 0 or i + 1 == num_frames:
                print("{} video: {}, frame: {}".format(hp.now_time(), video, i + 1))
        print('Finish video: {}.'.format(video))
        break
    print('Finish all.')
