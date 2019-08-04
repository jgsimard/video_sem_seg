import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import custom_transforms as tr
import torchvision.transforms as T
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from models.modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

from torchvision import transforms
from datasets.custom_transform_multichannel import ToNumpy, ToTensor, RandomDropOut, RandomNoise
from datasets.multiview_info import *

from models.modeling.unet_model import UNet
import albumentations as aug


def visualize(label, feature, num_classes, concat):
    plt.imshow(label.clone().cpu().data)
    plt.title('label')
    plt.show()
    if concat:
        for camid in range(CAM_NUM):
            for c in range(num_classes):
                plt.imshow(feature[camid * (num_classes + 1) + c, :, :].clone().cpu().data)
                plt.title('camera %d, projected feature %d' % (camid, c))
                plt.show()
            plt.imshow(feature[(camid + 1) * (num_classes + 1) - 1, :, :].clone().cpu().data, cmap='gray')
            plt.title('depth')
            plt.show()

    else:
        for camid in range(CAM_NUM):
            for c in range(num_classes):
                plt.imshow(feature[camid, c, :, :].clone().cpu().data)
                plt.title('camera %d, projected feature %d' % (camid, c))
                plt.show()
            plt.imshow(feature[camid, num_classes, :, :].clone().cpu().data, cmap='gray')
            plt.title('camera %d, depth' % camid)
            plt.show()


def project_point_cloud(xyz_pts, feature_src, cam_intrinsics_target, flip_horizontally=0, flip_vertically=1):
    """
    function to project a given feature vector to a target camera plane by using the point cloud

    :param xyz_pts:
        [B,3,W,H] tensor: point cloud points
    :param feature_src:
        [B,C+1,H,W]: features vector, including softmax scores of each class + depth
    :param cam_intrinsics_target:
        [3,3]: target camera intrinsics
    :param flip_horizontally
    :param flip_vertically

    :return:
        [B,C+1,H,W]: flipped feature vector
    """

    # flatten feature
    batch = feature_src.shape[0]
    channel = feature_src.shape[1]
    feature_src_flat = torch.flatten(feature_src, 2)

    # normalize depth
    xyz_pts = xyz_pts.view(batch, 3, -1)  # format it as dimension Bx3xN
    z = xyz_pts[:, 2, :].view(batch, 1, -1).expand(-1, 3, -1)
    xyz_pts_norm = (xyz_pts / z).type(torch.float)

    # project
    im_pts = torch.matmul(cam_intrinsics_target.cuda(), xyz_pts_norm)
    # round img pts and only take x,y
    im_pts = torch.round(im_pts)[:, 0:2, :]  # im_pts format is Bx2xN

    # find img within range
    valid_ind = (0 < im_pts[:, 0, :]) * (im_pts[:, 0, :] < WIDTH) * (0 < im_pts[:, 1, :]) * (
            im_pts[:, 1, :] < HEIGHT)  # valid_ind format BxN

    # generate feature vector
    feature_dst = torch.zeros_like(feature_src)
    for i in range(batch):
        im_pts_valid = im_pts[i, :, valid_ind[i, :]].type(torch.int64)  # im_pts_valid format 2xN

        # # find duplicates and choose the ones with smaller depth
        # vals, inverse, count = np.unique(im_pts_valid, return_inverse=True,
        #                               return_counts=True, axis=1)
        # idx_vals_repeated = np.where(count > 1)[0]
        # rows, cols = np.where(inverse == idx_vals_repeated[:, np.newaxis])
        # _, inverse_rows = np.unique(rows, return_index=True)
        # groups = np.split(cols, inverse_rows[1:])

        feature_src_valid = feature_src_flat[i, :, valid_ind[i, :]]  # feature_src_valid shape CxN

        feature_dst[i, :, im_pts_valid[1, :], im_pts_valid[0, :]] = feature_src_valid

    # reshape feature_dst to the same format as input
    feature_dst = feature_dst.view(batch, channel, HEIGHT, WIDTH)
    if flip_horizontally:
        feature_dst = torch.flip(feature_dst, [3])
    if flip_vertically:
        feature_dst = torch.flip(feature_dst, [2])

    return feature_dst


def transform_point_cloud(xyz_src, T_src, T_target):
    """
    transform xyz from src to target

    :param xyz_src:
        tensor [Bx3xHxW], src point cloud
    :param T_src:
        tensor [4x4], transformation from world to src camera
    :param T_target:
        tensor [4x4], transformation from world to target camera
    :return:
        transformed point cloud [Bx3xWxH]
    """

    batch = xyz_src.shape[0]

    # make it homogeneous
    xyz_homogeneous = torch.cat(
        (xyz_src.type(torch.float), torch.ones(batch, 1, HEIGHT, WIDTH).type(torch.float).cuda()),
        dim=1)  # shape Bx4xHxW
    xyz_homogeneous = xyz_homogeneous.view(batch, 4, -1).type(torch.float)  # shape Bx4xN

    # transform point cloud
    xyz_transformed_homogeneous = torch.matmul(torch.matmul(torch.inverse(T_target.cuda()), T_src.cuda()),
                                               xyz_homogeneous)

    # format into image shape
    xyz_transformed = xyz_transformed_homogeneous[:, 0:3, :].view(batch, 3, HEIGHT, WIDTH)  # shape Bx3xHxW

    return xyz_transformed


class Merger(nn.Module):
    """ Augmentation has to be put here due to projection"""

    def __init__(self, num_classes):
        super(Merger, self).__init__()
        self.num_classes = num_classes

        self.unet = UNet(n_channels=(self.num_classes + 1) * CAM_NUM, n_classes=self.num_classes)
        self._init_weight()

        self.transform_train = aug.Compose([
            aug.HorizontalFlip(p=0.5),
            aug.ShiftScaleRotate(p=0.7, rotate_limit=30, border_mode=0),
            aug.OneOf([aug.Blur(p=0.7, blur_limit=3), RandomNoise(p=0.7)]),
        ])

        self.scale = 10.0

    def forward(self, feature, xyzi, label):
        """
        project the images into one frame

        :param feature:
            [BxCAM_NUMxCxHxW]
        :param xyzi:
            [BxCAM_NUMx4xHxW]
        :param label:
            [BxCAM_NUMxHxW]
        :return:
        soft_label:
            B x CAM_NUM x NUM_CLASS x H x W
        """
        batch = feature.shape[0]
        soft_label = torch.empty(batch, CAM_NUM, self.num_classes, HEIGHT, WIDTH).cuda()
        z = xyzi[:, :, 2, :, :].float()

        for camid_target in range(CAM_NUM):
            # project all camera prediction to the target camera
            # the feature_tmp_projected will be B x CAM_NUM x (C+1) x H x W
            feature_all_projected = torch.zeros(
                [batch, CAM_NUM, self.num_classes + 1, HEIGHT, WIDTH]).cuda()  # BxCAM_NUMx(C+1)xHxW

            for camid_src in range(CAM_NUM):
                # combine feature and depth
                feature_src = torch.cat(
                    (feature[:, camid_src, :, :, :], z[:, camid_src, :, :].view(batch, 1, HEIGHT, WIDTH)),
                    dim=1)  # Bx(C+1)xHxW

                # if not the same camera
                if camid_src != camid_target:
                    # transform
                    xyz_tmp = transform_point_cloud(xyzi[:, camid_src, 0:3, :, :], T[camid_src], T[camid_target])
                    # project feature
                    feature_all_projected[:, camid_src, :, :, :] = project_point_cloud(xyz_tmp, feature_src,
                                                                                       CAM_MTX[camid_target])
                else:
                    feature_all_projected[:, camid_src, :, :, :] = feature_src

            # reorder feature (feature from target, i.e. itself, will always be first, the rest will be randomized)
            order = torch.randperm(CAM_NUM).cuda()
            order = order[order != camid_target]
            order = torch.cat((torch.tensor([camid_target]).cuda(), order))
            feature_all_projected = torch.index_select(feature_all_projected, 1, order)

            # # some visualization
            # visualize(label[0, camid_target, :, :], feature_all_projected[0, :, :, :, :], self.num_classes,
            #           concat=False)

            # augmentation and dropout
            if self.training:
                for b in range(batch):
                    # only transform the label for the current camera
                    curr_feature = feature_all_projected[b, :, :, :, :]
                    curr_label = label[b, camid_target, :, :].view(1, HEIGHT, WIDTH)
                    sample = ToNumpy({'image': curr_feature, 'label': curr_label})
                    sample = self.transform_train(image=sample['image'], mask=sample['label'])
                    sample = ToTensor(sample, batch=CAM_NUM, channel=self.num_classes + 1)
                    feature_all_projected[b, :, :, :, :] = sample['image']
                    label[b, camid_target, :, :] = sample['label']
                    # drop out
                    feature_all_projected[b, :, :, :, :] = RandomDropOut(feature_all_projected[b, :, :, :, :], CAM_NUM,
                                                                         p=0.0)
            # normalization
            feature_all_projected = feature_all_projected / self.scale

            # reformat the tensor to become BxCAM_NUM*(C+1)xHxW for network to process
            feature_target = feature_all_projected.view(batch, -1, HEIGHT, WIDTH).cuda()  # B x C x H x W

            # # some visualization
            # visualize(label[0, camid_target, :, :], feature_target[0, :, :, :], self.num_classes, concat=True)

            # go through network
            soft_label[:, camid_target, :, :, :] = self.unet(feature_target)

        return soft_label, label

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_merger(num_classes):
    return Merger(num_classes)


if __name__ == "__main__":
    from models.modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
    from datasets.isi_multiview import DeepSightDepthMultiview

    merger = build_merger(13)
    merger = merger.cuda()

    print("Testing Depth dataset")
    root_dir = "/home/deepsight3/dev/deepsight/MultiView/data"
    depth_dataset = DeepSightDepthMultiview(root_dir, split="train")
    train_generator = DataLoader(depth_dataset, shuffle=False, batch_size=2, num_workers=1)

    print(len(depth_dataset))
    plt.close()

    for i_batch, sample in enumerate(train_generator):
        img = sample['image']
        label = sample['label']
        pc = sample['pointcloud']

        print(img.shape, label.shape, pc.shape, np.unique(label.cpu().numpy()))

        img = img.cuda()
        label = label.cuda()
        pc = pc.cuda()

        feature_input = img[:, :, 0, :, :].view(2, 4, 1, HEIGHT, WIDTH)
        feature_input = feature_input.repeat(1, 1, 13, 1, 1)  # 13 times channels
        print(feature_input.shape)

        scores = merger(feature_input, pc, label)

        for s in scores:
            print(s.shape)

        break
