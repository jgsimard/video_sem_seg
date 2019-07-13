import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np

WIDTH = 352
HEIGHT = 287

CAM_NUM = 4

# camera serial number
# 0: 130600
# 1: 130602
# 2: 130603
# 3: 130604

# load intrinsics (should have been in a json/mat file)
CAM_MTX = CAM_NUM * [None]
CAM_MTX[0] = torch.tensor([[153.2559, 0, 179.5296],
                           [0, 153.2559, 135.8711],
                           [0, 0, 1.0000]]).type(torch.double)
CAM_MTX[1] = torch.tensor([[154.0505, 0, 176.8762],
                           [0, 154.0505, 141.7285],
                           [0, 0, 1.0000]]).type(torch.double)
CAM_MTX[2] = torch.tensor([[153.7142, 0, 180.1627],
                           [0, 153.7142, 143.3461],
                           [0, 0, 1.0000]]).type(torch.double)
CAM_MTX[3] = torch.tensor([[153.1698, 0, 178.6514],
                           [0, 153.1698, 138.2396],
                           [0, 0, 1.0000]]).type(torch.double)
# load distortion (should have been in a json/mat file)
CAM_DIST = CAM_NUM * [None]
CAM_DIST[0] = np.array([0.1466, -0.1815, -0.0022, 0.0000, 0.0447])
CAM_DIST[1] = np.array([0.1383, -0.1728, -0.0017, -0.0000, 0.0421])
CAM_DIST[2] = np.array([0.1498 - 0.1832, 0.4550 * 0.001, -0.2183 * 0.001, 0.0449])
CAM_DIST[3] = np.array([0.1470, -0.1806, -0.0022, 0.0013, 0.0441])

# load transformation (should have been in a json/mat file)
T = CAM_NUM * [None]
T[0] = torch.tensor([[0.998637149617664, 0.0210423473974330, 0.0477604754950395, -0.0123808505935194],
                     [0.0291696689445358, 0.533808124654113, -0.845102370406642, 1.18856957459405],
                     [-0.0432778675210864, 0.845343779576844, 0.532466825739935, 0.774820514239818],
                     [0, 0, 0, 1]]).type(torch.double)
T[1] = torch.tensor([[0.767718219791565, 0.474479685211881, 0.430671293820828, -0.585330929621780],
                     [0.0775255440921717, 0.598383956289324, -0.797449955087308, 1.30580527484513],
                     [-0.636080596318803, 0.645604886270830, 0.422605969917539, 0.319082229324211],
                     [0, 0, 0, 1]]).type(torch.double)
T[2] = torch.tensor([[0.779933518013299, -0.487300286641689, -0.392736728761557, 0.679696448874243],
                     [-0.00626475587245328, 0.621402929524694, -0.783466114144056, 1.24384315309774],
                     [0.625831015780825, 0.613511882376420, 0.481600155595578, 0.430077010957105],
                     [0, 0, 0, 1]]).type(torch.double)
T[3] = torch.tensor([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]).type(torch.double)


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
    xyz_pts_norm = (xyz_pts / z).type(torch.double)

    # project
    im_pts = torch.matmul(cam_intrinsics_target, xyz_pts_norm)
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
    xyz_homogeneous = torch.cat((xyz_src.type(torch.double), torch.ones(batch, 1, HEIGHT, WIDTH).type(torch.double)),
                                dim=1)  # shape Bx4xHxW
    xyz_homogeneous = xyz_homogeneous.view(batch, 4, -1).type(torch.double)  # shape Bx4xN

    # transform point cloud
    xyz_transformed_homogeneous = torch.matmul(torch.matmul(torch.inverse(T_target), T_src), xyz_homogeneous)

    # format into image shape
    xyz_transformed = xyz_transformed_homogeneous[:, 0:3, :].view(batch, 3, HEIGHT, WIDTH)  # shape Bx3xHxW

    return xyz_transformed


class Merger(nn.Module):
    def __init__(self, num_classes, BatchNorm):
        super(Merger, self).__init__()
        self.num_classes = num_classes

        self.conv = nn.Sequential(
            nn.Conv2d(CAM_NUM * (num_classes + 1), CAM_NUM * 16, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(CAM_NUM * 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(CAM_NUM * 16, CAM_NUM * 16, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(CAM_NUM * 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(CAM_NUM * 16, num_classes, kernel_size=1, stride=1))
        self._init_weight()

    def forward(self, feature, xyz, camid):
        """
        project the images into one frame

        :param feature:
            [BxCAM_NUMxCxHxW]
        :param xyz:
            [BxCAM_NUMx3xHxW]
        :param camid:
            [BxCAM_NUM]
        :return:
        """
        batch = feature.shape[0]

        # normalize feature
        feature = (feature - 0.5) / 0.5
        score = [None] * CAM_NUM
        for camid_target in range(CAM_NUM):
            # with torch.no_grad():
            feature_tmp_projected = torch.zeros(
                [batch, CAM_NUM, self.num_classes + 1, HEIGHT, WIDTH])  # BxCAM_NUMx(C+1)xHxW
            for camid_src in range(CAM_NUM):
                # combine feature and depth
                # TODO: verify this
                feature_tmp = torch.cat((feature[:, camid_src, :, :, :], xyz_tmp[:, camid_src, 2, :, :]),
                                        dim=1)  # BxCAM_NUMx(C+1)xHxW
                # if not the same camera
                if camid_src != camid_target:
                    # dropout
                    if torch.rand(1)[0] > 0.9:
                        continue
                    # transform
                    xyz_tmp = transform_point_cloud(xyz[:, camid_src, :, :, :], T[camid_src], T[camid_target])
                    # project feature
                    feature_tmp_projected[:, camid_target, :, :, :] = project_point_cloud(xyz_tmp, feature_tmp,
                                                                                          camid_target)
                else:
                    feature_tmp_projected[:, camid_target, :, :, :] = feature_tmp

            # reorder feature (self feature will always be first, the rest will be randomized)
            order = torch.randperm(CAM_NUM)
            order = order[order != camid_target]
            order = torch.cat((torch.tensor([camid_src]), order))
            feature_tmp_projected = torch.index_select(feature_tmp_projected, 1, order)

            # reshape
            feature_target = feature_tmp_projected.view(batch, -1, HEIGHT, WIDTH)
            score[camid_src] = self.conv(feature_target)

        return score

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


def build_merger(num_classes, BatchNorm):
    return Merger(num_classes, BatchNorm)


if __name__ == "__main__":
    print(1)
