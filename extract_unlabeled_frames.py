import os
import pprint
import cv2
from PIL import Image
import numpy as np

def video_id_to_name(datasets, sampeled_images_path):
    mapping = {}
    for dataset in datasets:
        for root, dirs, files in os.walk(os.path.join(sampeled_images_path, dataset)):
            if root.find("GH") >= 0:
                id = files[0].split("scene")[0]
                mapping[id] = root + ".MP4"
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
    # for id in mapping:
    id = '50'
    id_dir = os.path.join(output_dir, id)
    create_directory(id_dir)

    seq_dir = os.path.join(output_dir, id)
    labeled_rgb_images_dir = os.path.join(rgb_path, "RGB_Images", id)
    cap = cv2.VideoCapture(mapping[id])

    for labeled_rgb_image in os.listdir(labeled_rgb_images_dir):
        frame_id, _ = os.path.splitext(labeled_rgb_image)
        cap.set(1, int(frame_id))
        res, frame = cap.read()

        real_frame = np.asarray(Image.open(os.path.join(rgb_path, "RGB_Images", id, f"{frame_id}.jpg")))

        print(frame_id, int(frame_id), frame.shape, real_frame.shape, np.array_equal(real_frame, frame))


        cv2.imwrite(os.path.join(id_dir, f"{frame_id}.jpg"), frame)
        # cv2.imshow(frame)
        #
        # while True:
        #     ch = 0xFF & cv2.waitKey(1)  # Wait for a second
        #     if ch == 27:
        #         break




