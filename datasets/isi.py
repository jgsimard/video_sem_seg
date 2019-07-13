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
from models.modeling.merger import transform_point_cloud, project_point_cloud, CAM_MTX

WIDTH = 352
HEIGHT = 287

# load transformation (should have been in a json/mat file)
CAM_NUM = 4
T = CAM_NUM * [None]
T[0] = np.array([[0.998637149617664, 0.0210423473974330, 0.0477604754950395, -0.0123808505935194],
                 [0.0291696689445358, 0.533808124654113, -0.845102370406642, 1.18856957459405],
                 [-0.0432778675210864, 0.845343779576844, 0.532466825739935, 0.774820514239818],
                 [0, 0, 0, 1]])
T[1] = np.array([[0.767718219791565, 0.474479685211881, 0.430671293820828, -0.585330929621780],
                 [0.0775255440921717, 0.598383956289324, -0.797449955087308, 1.30580527484513],
                 [-0.636080596318803, 0.645604886270830, 0.422605969917539, 0.319082229324211],
                 [0, 0, 0, 1]])
T[2] = np.array([[0.779933518013299, -0.487300286641689, -0.392736728761557, 0.679696448874243],
                 [-0.00626475587245328, 0.621402929524694, -0.783466114144056, 1.24384315309774],
                 [0.625831015780825, 0.613511882376420, 0.481600155595578, 0.430077010957105],
                 [0, 0, 0, 1]])
T[3] = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])

CAMID = {'royale_20180717_130600': 0, 'royale_20180717_130602': 1, 'royale_20180717_130603': 2,
         'royale_20180717_130604': 3}


def pointcloud_reader(name, pointcloud_format='xyz', pointcloud_dtype=np.float32):
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
        self.transform_train = transforms.Compose([
            # tr.RandomHorizontalFlip(),
            #  tr.RandomScaleCrop(base_size=513, crop_size=513, fill=0),
            # tr.RandomRotate(15),
            # tr.RandomGaussianBlur(),
            # tr.Normalize(mean=(0.5,),
            #             std=(0.5,)),
            tr.ToTensor()])

        self.transform_validation = transforms.Compose([tr.Normalize(mean=(0.5, 0.5, 0.5),
                                                                     std=(0.5, 0.5, 0.5)),
                                                        tr.ToTensor()])

        print(self.xmls_dir)
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
            filename = "validation_tata.txt"
        file = open(filename, 'r').readlines()
        self.index = []
        for i in range(len(file)):
            self.index.append(int(i))
        # self.files = pd.DataFrame(list(zip(sequences, frame_ids)),
        #                          columns=['sequence', 'frame_id'])

    def __len__(self):
        return len(self.files[0])

    def __getitem__(self, item):
        index = self.index[item]

        img = np.zeros((CAM_NUM,) + (1, HEIGHT, WIDTH))
        depth = np.zeros((CAM_NUM,) + (1, HEIGHT, WIDTH))
        label = np.zeros((CAM_NUM,) + (HEIGHT, WIDTH))
        pointcloud = np.zeros((CAM_NUM,) + (3, HEIGHT, WIDTH))

        for seq in self.sequences:
            camid = CAMID[seq]
            frame = self.files[camid][index]
            print(frame)
            img_filename = os.path.join(self.images_dir, seq, frame + ".jpg")
            depth_filename = os.path.join(self.depths_dir, seq, frame + ".png")
            mask_filename = os.path.join(self.masks_dir, seq, frame + ".png")
            ply_filename = os.path.join(self.plys_dir, seq, frame + ".ply")

            img[camid, :, :, :] = np.asarray(Image.open(img_filename).convert('L'))[None,]
            depth[camid, :, :, :] = np.asarray(Image.open(depth_filename))[None,]
            label[camid, :, :] = Image.open(mask_filename)
            pointcloud_tmp = pointcloud_reader(ply_filename, pointcloud_format="xyz")
            # TODO: test point cloud
            pointcloud_tmp = linalg.inv(T[camid]) @ np.concatenate((pointcloud_tmp.T, np.ones((1, img[camid].size))))
            pointcloud[camid, :, :, :] = np.reshape(pointcloud_tmp[0:3, :], (3,) + (WIDTH, HEIGHT)).transpose(0, 2, 1)

            s = img[camid, :, :].shape
            print(s, s[0] * s[1])

        sample = {'image': img, 'depth': depth, 'label': label, 'pointcloud': pointcloud}

        # TODO: test aug
        if self.split == "train":
            sample['image'] = torch.tensor(sample['image'])
            sample['depth'] = torch.tensor(sample['depth'])
            sample['label'] = torch.tensor(sample['label'])
            sample['pointcloud'] = torch.tensor(sample['pointcloud'])

        #    return self.transform_train(sample)
        # elif self.split == 'validation':
        #    return self.transform_validation(sample)

        return sample


if __name__ == "__main__":
    print("Testing Depth dataset")
    root_dir = "/home/deepsight3/dev/deepsight/MultiView/data"
    depth_dataset = DeepSightDepth(root_dir, split="train")
    train_generator = DataLoader(depth_dataset, shuffle=False, batch_size=1, num_workers=1)

    print(len(depth_dataset))

    for i_batch, sample in enumerate(train_generator):
        img = sample['image']
        depth = sample['depth']
        label = sample['label']
        pc = sample['pointcloud']

        print(img.shape)
        print(depth.shape)
        print(img.shape)
        print(pc.shape)

        print(img.size, label.size, pc.shape, np.unique(label))

        plt.imshow(np.asarray(img[0, 0, 0, :, :]))
        plt.title("img")
        plt.show()

        plt.imshow(depth[0, 0, 0, :, :])
        plt.title("depth")
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
        feature_input = torch.cat((img, img))[:, src, :, :, :]
        xyz_input_transformed = transform_point_cloud(xyz_input, torch.tensor(T[src]), torch.tensor(T[target]))

        im_projected = project_point_cloud(xyz_input_transformed, feature_input, CAM_MTX[target])

        plt.imshow(im_projected[1, 0, :, :].reshape(HEIGHT, WIDTH), cmap='gray')
        plt.show()
        plt.imshow(img[0, src, 0, :, :], cmap='gray')
        plt.show()
        plt.imshow(img[0, target, 0, :, :], cmap='gray')
        plt.show()

        break
