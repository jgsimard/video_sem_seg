import os
from torch.utils.data import Dataset, DataLoader
from os.path import join
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from datasets import custom_transforms as tr
from plyfile import PlyData, PlyElement


class DeepSightRGB(Dataset):
    NUM_CLASSES = 11
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.imgs_dir = os.path.join(root_dir, "RGB_Images")
        self.masks_dir = os.path.join(root_dir, "Masks")
        self.transform_train =  transforms.Compose([tr.RandomHorizontalFlip(),
                                                    tr.RandomVerticalFlip(),
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


def pointcloud_reader(name, pointcloud_format='xyz',  pointcloud_dtype=np.float32):
    with open(name, 'rb') as f:
        plydata = PlyData.read(f)
    map_dict = dict(x='x', y='y', z='z', i='intensity')
    pointcloud = np.stack([plydata['vertex'][map_dict[c]].astype(pointcloud_dtype) for c in pointcloud_format]).T
    return pointcloud


class DeepSightDepth(Dataset):
    NUM_CLASSES = 11
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.depths_dir = os.path.join(root_dir, "Depths")
        self.images_dir = os.path.join(root_dir, "Images")
        self.masks_dir = os.path.join(root_dir, "Masks")
        self.overlays_dir = os.path.join(root_dir, "Overlays")
        self.plys_dir = os.path.join(root_dir, "PLYs")
        self.xmls_dir = os.path.join(root_dir, "XMLs")
        self.split = split
        self.transform_train =  transforms.Compose([tr.RandomHorizontalFlip(),
                                                    tr.RandomVerticalFlip(),
                                                    tr.RandomScaleCrop(base_size=513, crop_size=513, fill=0),
                                                    tr.RandomRotate(15),
                                                    tr.RandomGaussianBlur(),
                                                    tr.Normalize(mean=(0.5, 0.5, 0.5),
                                                                 std=(0.5, 0.5, 0.5)),
                                                    tr.ToTensor()])

        self.transform_validation = transforms.Compose([tr.Normalize(mean=(0.5, 0.5, 0.5),
                                                                     std=(0.5, 0.5, 0.5)),
                                                        tr.ToTensor()])

        print(self.xmls_dir)
        xml_files = []
        for dir_path, dir_names, file_names in os.walk(self.xmls_dir):
            xml_files += [os.path.join(dir_path, file) for file in file_names]
        sequences = [file.split("/")[-2] for file in xml_files]
        frame_ids = [file.split("/")[-1].split(".")[0] for file in xml_files]
        self.files = pd.DataFrame(list(zip(sequences, frame_ids)),
                                  columns=['sequence', 'frame_id'])


    def __len__(self):
        return self.files.shape[0]

    def __getitem__(self, item):
        basename = os.path.join(self.files['sequence'][item], self.files['frame_id'][item])
        img_filename = os.path.join(self.images_dir, basename + ".jpg")
        depth_filename = os.path.join(self.depths_dir, basename + ".png")
        mask_filename = os.path.join(self.masks_dir, basename + ".png")
        ply_filename = os.path.join(self.plys_dir, basename + ".ply")


        img = Image.open(img_filename)
        depth = Image.open(depth_filename)
        label = Image.open(mask_filename)
        pointcloud = pointcloud_reader(ply_filename, pointcloud_format="xyzi")
        pointcloud = np.reshape(pointcloud.T, (4,) + img.size).transpose((0, 2, 1))
        s = img.size
        print(s, s[0]*s[1])
        sample = {'image': img, 'depth': depth, 'label': label, 'pointcloud': pointcloud}

        if self.split == "train":
            return self.transform_train(sample)
        elif self.split == 'validation':
            return self.transform_validation(sample)

        return sample


if __name__ == "__main__":
    # print("Testing RGB dataset")
    # root_dir = "/home/deepsight2/development/data/rgb"
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
    #     if i == 5:
    #         break
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
    # root_dir = "/home/deepsight2/development/data/sem_seg"
    root_dir = "/home/deepsight2/development/data/sem_seg_07_10_2019"
    depth_dataset = DeepSightDepth(root_dir)
    print(len(depth_dataset))
    fig = plt.figure()
    for i in range(len(depth_dataset)):
        sample = depth_dataset[i]
        img = sample['image']
        depth = sample['depth']
        label = sample['label']
        pc = sample['pointcloud']

        print(i, img.size, label.size, pc.shape, np.unique(label))

        # plt.imshow(np.transpose(np.asarray(img), (1, 2, 0)))
        plt.imshow(np.asarray(img))
        plt.title("img")
        plt.show()

        plt.imshow(depth)
        plt.title("depth")
        plt.show()

        plt.imshow(sample['label'])
        plt.title("mask")
        plt.show()

        for j in range(4):
            plt.imshow(pc[j,:,:])
            plt.show()

        break


