import os
from torch.utils.data import Dataset, DataLoader
from os.path import join
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from datasets import custom_transforms as tr
from plyfile import PlyData
import cv2
import random



class DeepSightRGB(Dataset):
    NUM_CLASSES = 11
    def __init__(self, root_dir, split="train"):
        self.root_dir = root_dir
        self.split = split
        self.imgs_dir = os.path.join(root_dir, "RGB_Images")
        self.masks_dir = os.path.join(root_dir, "Masks")
        self.transform_train =  transforms.Compose([tr.RandomHorizontalFlip(),
                                                    tr.RandomScaleCrop(base_size=513, crop_size=513, fill=0),
                                                    tr.RandomRotate(15),
                                                    tr.RandomGaussianBlur(),
                                                    tr.Normalize(mean=(0.5, 0.5, 0.5),
                                                                 std=(0.5, 0.5, 0.5)),
                                                    tr.ToTensor()])

        self.transform_validation = transforms.Compose([tr.FixScaleCrop(crop_size=513),
                                                        tr.Normalize(mean=(0.5, 0.5, 0.5),
                                                                     std=(0.5, 0.5, 0.5)),
                                                        tr.ToTensor()])

        with open(os.path.join(root_dir, "Sets", f"{split}.txt"), 'r') as f:
            basenames, imgs, masks = [], [], []
            for basename in f:
                img, mask= self.get_filenames(basename)
                basenames.append(basename)
                imgs.append(img)
                masks.append(mask)
            self.files = pd.DataFrame(list(zip(basenames, imgs, masks)),
                                      columns=['basename', 'img_filename', 'label_filename'])
        # self.files = self.files[:100]

    def get_filenames(self, basename):
        scene, id_in_scene = basename.strip("\n").split("scene")
        img_filename = join(self.imgs_dir, scene, id_in_scene + ".jpg")
        mask_filename = join(self.masks_dir, scene, id_in_scene + ".png")
        return img_filename, mask_filename

    def __len__(self):
        return self.files.shape[0]

    def __getitem__(self, item):
        # img = np.asarray(Image.open(self.files['img_filename'][item]))
        # label = np.asarray(Image.open(self.files['label_filename'][item]))
        img = Image.open(self.files['img_filename'][item])
        label = Image.open(self.files['label_filename'][item])
        sample = {'image': img, 'label': label}

        if self.split == "train":
            return self.transform_train(sample)
        elif self.split == 'validation':
            return self.transform_validation(sample)
        return sample


def video_scene_to_name(datasets, sampeled_images_path):
    mapping = {}
    for dataset in datasets:
        for root, dirs, files in os.walk(os.path.join(sampeled_images_path, dataset)):
            if root.find("GH") >= 0:
                id = files[0].split("scene")[0]
                mapping[id] = root + ".MP4"
    return mapping

class DeepSightTemporalRGB(Dataset):
    NUM_CLASSES = 11

    def __init__(self, root_dir, split="train", sampeled_images_path="/home/deepsight/DeepSightData", train_range = 100, eval_distance = 5):
        self.root_dir = root_dir
        self.split = split
        self.imgs_dir = os.path.join(root_dir, "RGB_Images")
        self.masks_dir = os.path.join(root_dir, "Masks")
        self.train_range = train_range
        self.eval_distance = eval_distance

        self.datasets = ['Feb_11_CDE',  'Feb_8_CDE',  'March_1_CDE']
        self.mapping = video_scene_to_name(self.datasets, sampeled_images_path)

        self.transform_train =  transforms.Compose([tr.RandomHorizontalFlip(temporal=True),
                                                    tr.RandomScaleCrop(base_size=513, crop_size=513, fill=0, temporal=True),
                                                    tr.RandomRotate(15, temporal=True),
                                                    tr.RandomGaussianBlur(temporal=True),
                                                    tr.Normalize(mean=(0.5, 0.5, 0.5),
                                                                 std=(0.5, 0.5, 0.5), temporal=True),
                                                    tr.ToTensor(temporal=True)])

        self.transform_validation = transforms.Compose([tr.FixScaleCrop(crop_size=513, temporal=True),
                                                        tr.Normalize(mean=(0.5, 0.5, 0.5),
                                                                     std=(0.5, 0.5, 0.5), temporal=True),
                                                        tr.ToTensor(temporal=True)])

        with open(os.path.join(root_dir, "Sets", f"{split}.txt"), 'r') as f:
            scenes, ids = [], []
            for basename in f:
                scene, id = basename.strip("\n").split("scene")
                scenes.append(scene)
                ids.append(id)
            self.files = pd.DataFrame(list(zip(scenes, ids)),
                                      columns=['scene', 'id'])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        scene = self.files['scene'][item]
        id = self.files['id'][item]
        img_filename = join(self.imgs_dir, scene, id + ".jpg")
        label_filename = join(self.masks_dir, scene, id + ".png")
        img = Image.open(img_filename)
        label = Image.open(label_filename)

        flip = 1 <= int(scene) <= 12 \
               or 29 <= int(scene) <= 32 \
               or 42 <= int(scene) <= 54 \
               or int(scene) == 76 \
               or 78 <= int(scene) <= 79 \
               or int(scene) == 106 \
               or 109 <= int(scene) <= 114

        vid = cv2.VideoCapture(self.mapping[scene])
        fps = vid.get(cv2.CAP_PROP_FPS)
        frame_count = vid.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_pos = int(id) - random.randint(1, self.train_range) if self.split is "train" else self.eval_distance
        if fps > 31:
            frame_pos *= 2
        frame_pos = min(frame_count - 1, max(100, frame_pos)) # clip value in [100, frame_count - 1]
        vid.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        res, random_frame = vid.read()

        if flip:
            random_frame = cv2.flip(random_frame, 0)
            random_frame = cv2.flip(random_frame, 1)

        sample = {'image': img, 'label': label, "random_image" : random_frame}

        if self.split == "train":
            return self.transform_train(sample)
        elif self.split == 'validation':
            return self.transform_validation(sample)
        return sample




def pointcloud_reader(name, pointcloud_format='xyz',  pointcloud_dtype=np.float32):
    with open(name, 'rb') as f:
        plydata = PlyData.read(f)
    map_dict = dict(x='x', y='y', z='z', i='intensity')
    pointcloud = np.stack([plydata['vertex'][map_dict[c]].astype(pointcloud_dtype) for c in pointcloud_format]).T
    return pointcloud


class DeepSightDepth(Dataset):
    NUM_CLASSES = 13
    def __init__(self, root_dir, split="train", train_ratio = 0.9, transform=None, seed=1234):
        self.root_dir = root_dir
        self.depths_dir = os.path.join(root_dir, "Depths")
        self.images_dir = os.path.join(root_dir, "Images")
        self.masks_dir = os.path.join(root_dir, "Masks")
        self.overlays_dir = os.path.join(root_dir, "Overlays")
        self.plys_dir = os.path.join(root_dir, "PLYs")
        self.xmls_dir = os.path.join(root_dir, "XMLs")
        self.split = split
        self.weights = [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1]

        self.transform_train =  transforms.Compose([tr.RandomHorizontalFlip(),
                                                    tr.RandomScaleCrop(base_size=(287, 352), crop_size=(287, 352), fill=0),
                                                    tr.RandomRotate(15),
                                                    tr.RandomGaussianBlur(),
                                                    tr.Normalize(mean=(0.5, 0.5, 0.5),
                                                                 std=(0.5, 0.5, 0.5)),
                                                    tr.ToTensor()])

        self.transform_validation = transforms.Compose([tr.Normalize(mean=(0.5, 0.5, 0.5),
                                                                     std=(0.5, 0.5, 0.5)),
                                                        tr.ToTensor()])

        xml_files = []
        for dir_path, dir_names, file_names in os.walk(self.xmls_dir):
            xml_files += [os.path.join(dir_path, file) for file in file_names]
        sequences = [file.split("/")[-2] for file in xml_files]
        frame_ids = [file.split("/")[-1].split(".")[0] for file in xml_files]
        self.files = pd.DataFrame(list(zip(sequences, frame_ids)),
                                  columns=['sequence', 'frame_id'])

        # self.files = self.files[:100]
        np.random.seed(seed)
        mask = np.random.rand(len(self.files)) < train_ratio
        if split == "train":
            self.files = self.files[mask]
        else:
            self.files = self.files[~mask]
        self.files = self.files.reset_index()



    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        # print(item)
        # try:
        basename = os.path.join(self.files['sequence'][item], self.files['frame_id'][item])
        # except:
        #     print(f"ERROR : item={item}")

        img_filename = os.path.join(self.images_dir, basename + ".jpg")
        depth_filename = os.path.join(self.depths_dir, basename + ".png")
        mask_filename = os.path.join(self.masks_dir, basename + ".png")
        ply_filename = os.path.join(self.plys_dir, basename + ".ply")


        img = Image.open(img_filename)
        depth = Image.open(depth_filename)
        label = np.array(Image.open(mask_filename))
        label[label==255]=0
        label = Image.fromarray(label)
        pointcloud = pointcloud_reader(ply_filename, pointcloud_format="xyzi")
        pointcloud = np.reshape(pointcloud.T, (4,) + img.size).transpose((0, 2, 1))
        s = img.size
        sample = {'image': img, 'depth': depth, 'label': label, 'pointcloud': pointcloud}

        if self.split == "train":
            return self.transform_train(sample)
        elif self.split == 'validation':
            return self.transform_validation(sample)

        return sample

if __name__ == "__main__":
    print("Testing RGB dataset")
    # root_dir = "/home/deepsight/data/rgb"
    # set = "validation"
    # rgb_dataset = DeepSightRGB(root_dir, set)
    # fig = plt.figure()
    # print(len(rgb_dataset))
    # for i in range(len(rgb_dataset)):
    #     sample = rgb_dataset[i]
    #     img = sample['image']
    #     label = sample['label']
    #     print(i, img.shape, label.shape, np.unique(label))
    #
    #     plt.imshow(np.transpose(np.asarray(img), (1, 2, 0)))
    #     plt.show()
    #     plt.imshow(sample['label'])
    #     plt.show()
    #
    #     break
    #
    # rgb_dataset = DeepSightRGB(root_dir, set)
    #
    # dataloader = DataLoader(rgb_dataset,
    #                         batch_size=4,
    #                         shuffle=True,
    #                         num_workers=4)
    #
    # for i_batch, sample_batched in enumerate(dataloader):
    #     print(sample_batched['image'].shape)
    #     break

    print("Testing Depth dataset")
    # # root_dir = "/home/deepsight/data/sem_seg_07_10_2019"
    # root_dir = "/home/deepsight/data/sem_seg_multiview_07_10_2019"
    # # split = "validation"
    # split = "train"
    # depth_dataset = DeepSightDepth(root_dir, split)
    # print(len(depth_dataset))
    # fig = plt.figure()
    # n = np.zeros(13)
    # for i in range(len(depth_dataset)):
    #     sample = depth_dataset[i]
    #     # print(sample)
    #     img = sample['image']
    #     # depth = sample['depth']
    #     label = sample['label']
    #     # pc = sample['pointcloud']
    #     uniques = np.unique(label).astype(int)
    #     n[uniques] +=1
    #     print(n)
    #     # print(i, img.size(), label.size(), uniques)
    #     # print(i, img.size, label.size, pc.shape, np.unique(label))
    #     print(np.asarray(img).min(), np.asarray(img).max())
    #
    #     # plt.imshow(np.transpose(np.asarray(img), (1, 2, 0)))
    #     # plt.imshow(np.asarray(img))
    #     plt.imshow(np.asarray(img)[0,:,:], cmap='gray')
    #     plt.title("img")
    #     plt.show()
    #     #
    #     # # plt.imshow(depth)
    #     # # plt.title("depth")
    #     # # plt.show()
    #     #
    #     plt.imshow(sample['label'])
    #     plt.title("mask")
    #     plt.show()
    #
    #     # for j in range(4):
    #     #     plt.imshow(pc[j,:,:])
    #     #     plt.show()
    #     #
    #     # depth_array = np.asarray(depth)
    #     # print(np.asarray(label).max())
    #     break

    print("Testing Temporal RGB dataset")
    root_dir = "/home/deepsight/data/rgb"
    set = "train"
    rgb_dataset = DeepSightTemporalRGB(root_dir, set)
    fig = plt.figure()
    print(len(rgb_dataset))
    for i in range(len(rgb_dataset)):
        sample = rgb_dataset[i]
        img = sample['image']
        label = sample['label']
        print(i, img.shape, label.shape, np.unique(label))


