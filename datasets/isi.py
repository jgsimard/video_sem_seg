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


def ply_pointcloud_writer(name, dt,
                          pointcloud_format='xyz',
                          semantic_seg=None,
                          scores=None,
                          instance_seg=None,
                          class_label=None):
    assert (pointcloud_format == 'xyz'), 'Only supports xyz format.'

    # Add the point clould
    vertex = np.array([tuple(r) for r in dt],
                      dtype=[('x', 'd'), ('y', 'd'), ('z', 'd')])

    ply_elem_list = [PlyElement.describe(vertex, 'vertex')]

    # Add pixel level labels
    vertex_label_list = []
    vertex_label_dtype_list = []

    if semantic_seg is not None:
        vertex_label_list.append(semantic_seg)
        vertex_label_dtype_list.append(('class', 'i4'))

    if scores is not None:
        vertex_label_list.append(scores)
        vertex_label_dtype_list.append(('score', 'd'))

    if instance_seg is not None:
        vertex_label_list.append(instance_seg)
        vertex_label_dtype_list.append(('instance_id', 'i4'))

    if vertex_label_list is not None:
        vertex_label = np.array(zip(*vertex_label_list),
                                dtype=vertex_label_dtype_list)
        ply_elem_list.append(PlyElement.describe(vertex_label, 'vertex_label'))

    # Add class label
    if class_label is not None:
        label = np.array([(class_label)], dtype=[('class_id', 'i4')])
        ply_elem_list.append(PlyElement.describe(label, 'label'))

    PlyData(ply_elem_list, text=True).write(name)

def pointcloud_reader(name, pointcloud_format='xyz',
                          pointcloud_dtype=np.float32):
    with open(name, 'rb') as f:
        plydata = PlyData.read(f)
    map_dict = dict(x='x', y='y', z='z', i='intensity')
    pointcloud = np.stack([plydata['vertex'][map_dict[c]].astype(pointcloud_dtype) for c in pointcloud_format]).T
    return pointcloud

def ply_pointcloud_reader(name, pointcloud_format='xyz',
                          load_semantic_seg=False,
                          load_instance_seg=False,
                          load_scores=False,
                          load_class_label=True,
                          pointcloud_dtype=np.float32,
                          npoints=None):
    try:
        with open(name, 'rb') as f:
            plydata = PlyData.read(f)

        map_dict = dict(x='x', y='y', z='z', i='intensity')

        col_list = []
        for c in pointcloud_format:
            col = plydata['vertex'][map_dict[c]].astype(pointcloud_dtype)
            col_list.append(col)
        # pointcloud = np.unique(np.stack(col_list).T, axis=0)
        pointcloud = np.stack(col_list).T

        dt = [pointcloud]

        if load_semantic_seg:
            dt.append(plydata['vertex_label']['class'].astype(np.int32))

        if load_instance_seg:
            dt.append(plydata['vertex_label']['instance_id'].astype(np.int32))

        if load_scores:
            dt.append(plydata['vertex_label']['score'].astype(np.float32))

        # Subsample points
        if npoints:
            while dt[0].shape[0] < npoints:
                dt = [np.concatenate((e, e)) for e in dt]

            choice = np.random.choice(dt[0].shape[0], size=npoints, replace=False)
            dt = [e[choice] for e in dt]

        if load_class_label:
            dt.append(int(plydata['label']['class_id'][0]))
    except ValueError as e:
        print('Error in reading {}'.format(name))
        raise
    return [dt]

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
        # self.transform_train =  transforms.Compose([tr.RandomHorizontalFlip(),
        #                                             tr.RandomVerticalFlip(),
        #                                             tr.RandomScaleCrop(base_size=513, crop_size=513, fill=0),
        #                                             tr.RandomRotate(15),
        #                                             tr.RandomGaussianBlur(),
        #                                             tr.Normalize(mean=(0.5, 0.5, 0.5),
        #                                                          std=(0.5, 0.5, 0.5)),
        #                                             tr.ToTensor()])
        #
        # self.transform_validation = transforms.Compose([tr.Normalize(mean=(0.5, 0.5, 0.5),
        #                                                              std=(0.5, 0.5, 0.5)),
        #                                                 tr.ToTensor()])

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
        print(ply_filename)
        print(pointcloud)
        print(pointcloud.shape)
        s = img.size
        print(s, s[0]*s[1])
        sample = {'image': img, 'label': label}



        # if self.split == "train":
        #     return self.transform_train(sample)
        # elif self.split == 'validation':
        #     return self.transform_validation(sample)
        # return sample


if __name__ == "__main__":
    # print("Testing RGB dataset")
    # root_dir = "/home/deepsight2/development/data/rgb"
    # set = "validation"
    # rgb_dataset = DeepSightRGB(root_dir, set)
    # fig = plt.figure()
    # print(len(rgb_dataset))
    # for i in range(len(rgb_dataset)):
    #     sample = rgb_dataset[i]
    #
    #     print(i, sample['image'].shape, sample['label'].shape)
    #
    #     plt.imshow(np.transpose(np.asarray(sample['image']), (1, 2, 0)))
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
    root_dir = "/home/deepsight2/development/data/sem_seg"
    depth_dataset = DeepSightDepth(root_dir)
    print(len(depth_dataset))
    fig = plt.figure()
    for i in range(len(depth_dataset)):
        sample = depth_dataset[i]

        print(i, sample['image'].shape, sample['label'].shape)

        plt.imshow(np.transpose(np.asarray(sample['image']), (1, 2, 0)))
        plt.show()
        plt.imshow(sample['label'])
        plt.show()

        if i == 5:
            break


