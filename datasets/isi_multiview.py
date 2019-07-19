import os
from torch.utils.data import Dataset, DataLoader
from os.path import join
import pandas as pd
from PIL import Image
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from torchvision import transforms
from datasets import custom_transforms as tr
from plyfile import PlyData, PlyElement
import torch
import cv2

from datasets.isi import DeepSightDepth
from datasets.multiview_info import *


def pointcloud_reader(name, pointcloud_format='xyz', pointcloud_dtype=np.float32):
    with open(name, 'rb') as f:
        plydata = PlyData.read(f)
    map_dict = dict(x='x', y='y', z='z', i='intensity')
    pointcloud = np.stack([plydata['vertex'][map_dict[c]].astype(pointcloud_dtype) for c in pointcloud_format]).T
    return pointcloud


class DeepSightDepthMultiview(Dataset):
    NUM_CLASSES = 13

    def __init__(self, root_dir, split="train", transform=None, seed=1234):
        self.root_dir = root_dir
        self.depths_dir = os.path.join(root_dir, "Depths")
        self.images_dir = os.path.join(root_dir, "Images")
        self.masks_dir = os.path.join(root_dir, "Masks")
        self.overlays_dir = os.path.join(root_dir, "Overlays")
        self.plys_dir = os.path.join(root_dir, "PLYs")
        self.xmls_dir = os.path.join(root_dir, "XMLs")
        self.split = split
        self.weights = [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1]
        np.random.seed(seed)

        self.transform = transforms.Compose([
            tr.Normalize(mean=(0.5, 0.5, 0.5),
                         std=(0.5, 0.5, 0.5),
                         multiview=True),
            tr.ToTensor(multiview=True)])

        # find the sequences
        self.sequences = os.listdir(self.xmls_dir)
        self.sequences.sort()

        # read the frames
        self.files = [[], [], [], []]
        for idx, seq in enumerate(self.sequences):
            self.files[idx] = os.listdir(os.path.join(self.xmls_dir, seq))
            self.files[idx].sort()
            self.files[idx] = [file.split('.')[0] for file in self.files[idx]]

        # read data split index
        if self.split == "train":
            filename = "train_data.txt"
        elif self.split == "validation":
            filename = "validation_data.txt"
        file = open(os.path.join(self.root_dir, filename), 'r').readlines()

        # get indexes
        self.index = []
        for i in range(len(file)):
            self.index.append(int(i))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        index = self.index[item]

        img = torch.zeros((CAM_NUM,) + (3, HEIGHT, WIDTH))
        label = torch.zeros((CAM_NUM,) + (HEIGHT, WIDTH))
        pointcloud = torch.zeros((CAM_NUM,) + (4, HEIGHT, WIDTH))

        for seq in self.sequences:
            camid = CAMID[seq]
            frame = self.files[camid][index]
            img_filename = os.path.join(self.images_dir, seq, frame + ".jpg")
            mask_filename = os.path.join(self.masks_dir, seq, frame + ".png")
            ply_filename = os.path.join(self.plys_dir, seq, frame + ".ply")

            # read image
            img_tmp = Image.open(img_filename)
            label_tmp = np.array(Image.open(mask_filename))
            label_tmp[label_tmp == 255] = 0
            label_tmp = Image.fromarray(label_tmp)

            pointcloud_tmp = pointcloud_reader(ply_filename, pointcloud_format="xyzi")
            pointcloud_tmp = np.reshape(pointcloud_tmp.T, (4,) + (WIDTH, HEIGHT))

            sample = {'image': img_tmp, 'label': label_tmp, 'pointcloud': pointcloud_tmp}
            sample = self.transform(sample)

            img[camid, :, :, :] = sample['image']
            label[camid, :, :] = sample['label']
            pointcloud[camid, :, :, :] = sample['pointcloud']

        return {'image': img, 'label': label, 'pointcloud': pointcloud}

    # def __getitem__(self, item):
    #     index = self.index[item]
    #
    #     img = torch.zeros((CAM_NUM,) + (3, HEIGHT, WIDTH))
    #     label = torch.zeros((CAM_NUM,) + (HEIGHT, WIDTH))
    #     pointcloud = torch.zeros((CAM_NUM,) + (3, HEIGHT, WIDTH))
    #
    #     # pointcloud[camid, :, :, :] = np.reshape(pointcloud_tmp[0:3, :], (3,) + (WIDTH, HEIGHT)).transpose(0, 2, 1)
    #     # img[camid, :, :, :] = np.array().transpose(2, 0, 1).astype(np.float32)[None,]
    #     # depth[camid, :, :, :] = np.asarray(Image.open(depth_filename)).astype(np.float32)[None,]
    #
    #     for seq in self.sequences:
    #         camid = CAMID[seq]
    #         frame = self.files[camid][index]
    #         img_filename = os.path.join(self.images_dir, seq, frame + ".jpg")
    #         depth_filename = os.path.join(self.depths_dir, seq, frame + ".png")
    #         mask_filename = os.path.join(self.masks_dir, seq, frame + ".png")
    #         ply_filename = os.path.join(self.plys_dir, seq, frame + ".ply")
    #
    #         # read image
    #         img_tmp = Image.open(img_filename)
    #         label_tmp = np.array(Image.open(mask_filename))
    #         label_tmp[label_tmp == 255] = 0
    #         label_tmp = Image.fromarray(label_tmp)
    #
    #         pointcloud_tmp = pointcloud_reader(ply_filename, pointcloud_format="xyz")
    #         pointcloud_tmp = linalg.inv(T[camid].cpu().numpy()) @ np.concatenate(
    #             (pointcloud_tmp.T, np.ones((1, HEIGHT * WIDTH))))
    #         pointcloud_tmp = np.reshape(pointcloud_tmp[0:3, :], (3,) + (WIDTH, HEIGHT))
    #
    #         sample = {'image': img_tmp, 'label': label_tmp, 'pointcloud': pointcloud_tmp}
    #         sample = self.transform(sample)
    #
    #         img[camid, :, :, :] = sample['image']
    #         label[camid, :, :] = sample['label']
    #         pointcloud[camid, :, :, :] = sample['pointcloud']
    #
    #         # TODO: undistort image
    #         # img_tmp = Image.open(img_filename).convert('L')
    #         # img_tmp = cv2.undistort(np.array(img_tmp), CAM_MTX[camid].numpy(), CAM_DIST[camid], None, None)
    #         # img[camid, :, :, :] = img_tmp[None,]
    #         #
    #         # depth_tmp = Image.open(depth_filename)
    #         # depth_tmp = cv2.undistort(np.array(depth_tmp), CAM_MTX[camid].numpy(), CAM_DIST[camid], None, None)
    #         # depth[camid, :, :, :] = np.asarray(depth_tmp)[None,]
    #         #
    #         # label_tmp = Image.open(mask_filename)
    #         # label_tmp = cv2.undistort(np.array(label_tmp), CAM_MTX[camid].numpy(), CAM_DIST[camid], None, None)
    #         # label[camid, :, :] = label_tmp
    #
    #     return {'image': img, 'label': label, 'pointcloud': pointcloud}


if __name__ == "__main__":
    from models.modeling.merger import transform_point_cloud, project_point_cloud

    print("Testing Depth dataset")
    root_dir = "/home/deepsight3/dev/deepsight/MultiView/data"
    depth_dataset = DeepSightDepthMultiview(root_dir, split="train")
    train_generator = DataLoader(depth_dataset, shuffle=False, batch_size=1, num_workers=1)

    print(len(depth_dataset))

    for i_batch, sample in enumerate(train_generator):
        img = sample['image']
        label = sample['label']
        pc = sample['pointcloud']

        print(img.shape)
        print(img.shape)
        print(pc.shape)

        print(img.size, label.size, pc.shape, np.unique(label))

        plt.imshow(np.asarray(img[0, 0, 0, :, :]))
        plt.title("img")
        plt.show()

        plt.imshow(label[0, 0, :, :])
        plt.title("mask")
        plt.show()

        for j in range(3):
            plt.imshow(pc[0, 0, j, :, :])
            plt.show()

        # test projection
        src = 0
        target = 1

        xyz_input = torch.cat((pc, pc))[:, src, :, :, :]
        feature_input = torch.cat((img[:, :, 0, :, :], img[:, :, 0, :, :]))[:, src, :, :].view(2, 1, HEIGHT, WIDTH)
        xyz_input_transformed = transform_point_cloud(xyz_input, torch.tensor(T[src]), torch.tensor(T[target]))

        im_projected = project_point_cloud(xyz_input_transformed, feature_input, CAM_MTX[target])

        plt.imshow(im_projected[1, 0, :, :].reshape(HEIGHT, WIDTH), cmap='gray')
        plt.show()
        plt.imshow(img[0, src, 0, :, :], cmap='gray')
        plt.show()
        plt.imshow(img[0, target, 0, :, :], cmap='gray')
        plt.show()

        break