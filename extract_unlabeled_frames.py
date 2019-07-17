import os
import pprint
import cv2
from PIL import Image
import numpy as np
import imageio

def video_id_to_name(datasets, sampeled_images_path):
    mapping = {}
    for dataset in datasets:
        for root, dirs, files in os.walk(os.path.join(sampeled_images_path, dataset)):
            if root.find("GH") >= 0:
                id = files[0].split("scene")[0]
                mapping[id] = root + ".MP4"
                # video_path = root + ".MP4"
                # vid = cv2.VideoCapture(video_path)
                # fps = vid.get(cv2.CAP_PROP_FPS)
                # mapping[id] = {"path": video_path, "fps" : fps}
    return mapping


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == "__main__":
    rgb_path = "/home/deepsight/data/rgb/"
    output_dir = f"{rgb_path}/unlabeled_rgb_images"

    datasets = ['Feb_11_CDE',  'Feb_8_CDE',  'March_1_CDE']
    sampeled_images_path = "/home/deepsight/DeepSightData"
    print("Extracting video_id_to_name...")
    mapping = video_id_to_name(datasets, sampeled_images_path)

    pprint.PrettyPrinter().pprint(mapping)
    expected_number_of_vid = 120
    id_for_all = sum([int(k) for k in mapping.keys()]) == expected_number_of_vid * (expected_number_of_vid + 1) / 2
    print(f"All Ids have a video = {id_for_all}")

    print("Extracting unlabeled images...")

    create_directory(output_dir)

    for id in sorted(mapping.keys()):
        print(id)
        flip = 1 <= int(id) <= 12 \
               or 29 <= int(id) <= 32 \
               or 42 <= int(id) <= 54 \
               or int(id)== 76 \
               or 78 <= int(id) <= 79 \
               or int(id) == 106 \
               or 109 <= int(id) <= 114
        id_dir = os.path.join(output_dir, id)
        create_directory(id_dir)

        seq_dir = os.path.join(output_dir, id)
        labeled_rgb_images_dir = os.path.join(rgb_path, "RGB_Images", id)
        vid = cv2.VideoCapture(mapping[id])
        fps = vid.get(cv2.CAP_PROP_FPS)
        frame_count = vid.get(cv2.CAP_PROP_FRAME_COUNT)
        print(f"id = {id}, file={mapping[id]}, Frame count= {frame_count}, FPS={int(fps)}")
        # vid.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
        # print(vid.get(cv2.CAP_PROP_POS_AVI_RATIO))
        # vid = imageio.get_reader(mapping[id]["path"], 'ffmpeg')

        frame_ids = sorted([os.path.splitext(img)[0] for img in  os.listdir(labeled_rgb_images_dir)])
        for frame_id in frame_ids:
            for distance in [2, 5, 10, 20, 30, 40 ,50, 60, 70, 80]:
                frame_pos = int(frame_id) - distance
                new_frame_path = os.path.join(id_dir, f"{frame_pos}.jpg")
                if os.path.isfile(new_frame_path):
                    continue
                pos_type = cv2.CAP_PROP_POS_FRAMES
                frame_pos = max(10, frame_pos)
                if fps > 31:
                    frame_pos *= 2

                frame_pos = min(frame_count - 1, max(100, frame_pos))
                vid.set(pos_type, frame_pos)
                res = False
                while res == False:
                    res, frame = vid.read()

                if flip:
                    frame = cv2.flip(frame, 0)
                    frame = cv2.flip(frame, 1)

                # frame = vid.get_data(frame_pos)
                # print(res)
                print(frame_id, frame_pos, int(vid.get(cv2.CAP_PROP_POS_MSEC)), frame.shape)


                cv2.imwrite(new_frame_path, frame)
                # cv2.imshow(frame_id, cv2.resize(frame, (384, 288)))

                # while True:
                #     ch = 0xFF & cv2.waitKey(0.01)  # Wait for a second
                #     if ch == 27:
                #         break
        vid.release()


