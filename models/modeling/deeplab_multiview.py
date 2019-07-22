import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from models.modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.modeling.deeplab import DeepLab
from models.modeling.merger import build_merger
from models.modeling.aspp import build_aspp
from models.modeling.decoder import build_decoder
from models.modeling.backbone import build_backbone
from datasets.utils import decode_seg_map_sequence
from datasets.multiview_info import *
import matplotlib.pyplot as plt


def visualize(image, soft_label):
    # some visualization
    grid_image = make_grid(image[:4].clone().cpu().data, 4, normalize=True)
    plt.imshow(np.transpose(grid_image, (1, 2, 0)))
    plt.show()
    hard_label = torch.max(soft_label[:4], 1)[1]
    grid_image = make_grid(
        decode_seg_map_sequence(hard_label.clone().cpu().numpy(),
                                dataset='isi_multiview'), 4, normalize=False, range=(0, 255))
    plt.imshow(np.transpose(grid_image, (1, 2, 0)))
    plt.show()
    # for i in range(CAM_NUM):
    #     plt.imshow(soft_label[i, 0, :, :].clone().cpu().numpy())
    #     plt.show()
    #     plt.imshow(soft_label[i, 1, :, :].clone().cpu().numpy())
    #     plt.show()
    #     plt.imshow(soft_label[i, 2, :, :].clone().cpu().numpy())
    #     plt.show()
    #     plt.imshow(soft_label[i, 3, :, :].clone().cpu().numpy())
    #     plt.show()
    #     plt.imshow(soft_label[i, 4, :, :].clone().cpu().numpy())
    #     plt.show()
    #     plt.imshow(soft_label[i, 5, :, :].clone().cpu().numpy())
    #     plt.show()
    #     plt.imshow(soft_label[i, 6, :, :].clone().cpu().numpy())
    #     plt.show()
    #     plt.imshow(soft_label[i, 7, :, :].clone().cpu().numpy())
    #     plt.show()
    #     plt.imshow(soft_label[i, 8, :, :].clone().cpu().numpy())
    #     plt.show()
    #     plt.imshow(soft_label[i, 9, :, :].clone().cpu().numpy())
    #     plt.show()
    #     plt.imshow(soft_label[i, 10, :, :].clone().cpu().numpy())
    #     plt.show()
    #     plt.imshow(soft_label[i, 11, :, :].clone().cpu().numpy())
    #     plt.show()
    #     plt.imshow(soft_label[i, 12, :, :].clone().cpu().numpy())
    #     plt.show()


class DeepLabMultiView(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(DeepLabMultiView, self).__init__()

        self.deeplab = DeepLab(backbone=backbone, output_stride=output_stride,
                               num_classes=num_classes, sync_bn=sync_bn,
                               freeze_bn=freeze_bn)

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.merger = build_merger(num_classes, BatchNorm)

        self.num_classes = num_classes

    def forward(self, image, pointcloud, label):  # x
        """

        :param image:
            B x CAM_NUM x 3 x H x W
        :param pointcloud:
            B x CAM_NUM x 4 x H x W
        :param label:
            B x CAM_NUM x H x W
        :return:
        soft_label:
            B x CAM_NUM x NUM_CLASS x H x W
        """
        # image = x[:, :, :3]
        # pointcloud = x[:, :, 3:6]
        # label = x[:, :, 6, :, :]
        # print(image.shape, pointcloud.shape, label.shape)

        batch = image.shape[0]

        # print('printing maximum and minimum z')
        # print(pointcloud.shape)
        # print(pointcloud[:, :, 2, :, :].max())
        # print(pointcloud[:, :, 2, :, :].min())
        # plt.imshow(pointcloud[0, 0, 2, :, :].clone().cpu().data)
        # plt.show()
        # print('printing maximum and minimum z')

        # predict for each camera
        soft_label_singleview = torch.empty(batch, CAM_NUM, self.num_classes, HEIGHT, WIDTH).cuda()
        for b in range(batch):
            x = self.deeplab(image[b, :, :, :, :])  # CAM_NUM x num_class x H x W
            # print(x.max())
            # print(x.min())
            soft_label_singleview[b, :, :, :, :] = x
            # visualize(image[b, :, :, :, :], x)

        # merge the labels
        soft_label_multiview = self.merger(soft_label_singleview, pointcloud, label)

        return soft_label_singleview, soft_label_multiview


if __name__ == "__main__":
    model = DeepLabMultiView(backbone='resnet', output_stride=16, num_classes=13)
    model.eval()
    batch = 1
    image = torch.rand(batch, 4, 3, 287, 352)
    pointcloud = torch.rand(batch, 4, 3, 287, 352)
    label = torch.randint(high=12, size=(batch, 4, 287, 352))
    output = model(image, pointcloud, label)

    for o in output:
        print(o.size())
