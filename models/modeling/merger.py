import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

WIDTH = 352
HEIGHT = 287

CAM_NUM = 4

# load intrinsics (should have been in a json/mat file)
CAM_MTX = CAM_NUM * [None]
CAM_MTX[0] = torch.tensor([[153.9002, 0, 176.2514],
                           [0, 153.9002, 139.8164],
                           [0, 0, 1.0000]]).type(torch.double)
CAM_MTX[1] = torch.tensor([[154.0505, 0, 176.876],
                           [0, 154.0505, 141.7285],
                           [0, 0, 1.0000]])
CAM_MTX[2] = torch.tensor([[153.7142, 0, 180.1627],
                           [0, 153.7142, 143.3461],
                           [0, 0, 1.0000]])
CAM_MTX[3] = torch.tensor([[153.2559, 0, 179.5296],
                           [0, 153.2559, 135.8711],
                           [0, 0, 1.0000]])
# load distortion (should have been in a json/mat file)
CAM_DIST = CAM_NUM * [None]
CAM_DIST[0] = np.array([0.1354, -0.1671, -0.7283 * 0.001, 0.5925 * 0.001, 0.0401])
CAM_DIST[1] = np.array([0.1383, -0.1728, -0.0017, -0.0000, 0.0421])
CAM_DIST[2] = np.array([0.1498 - 0.1832, 0.4550 * 0.001, -0.2183 * 0.001, 0.0449])
CAM_DIST[3] = np.array([0.1466, -0.1815, -0.0022, 0.0000, 0.0447])

# load transformation (should have been in a json/mat file)
T = CAM_NUM * [None]
T[0] = np.array([[0.999922316117099, 0.00641768054260757, -0.0106852752641006, 0.00724465381427334],
                 [-0.000695814985233705, -0.827184945604821, -0.561929338623867, -0.0460028374665461],
                 [-0.0124449818209144, 0.561893120745577, -0.827116221148077, 0],
                 [0, 0, 0, 1]])
T[1] = np.array([[0.999907288140456, 0.00819807594003977, -0.0108723497415904, -0.0294626922155813],
                 [0.0135237145816606, -0.691056494645518, 0.722674220510264, 0.00273410220614242],
                 [-0.00158886961349308, -0.722754249401364, -0.691103299451859, -0.132442759176783],
                 [0, 0, 0, 1]])
T[2] = np.array([[0.994721936649566, -0.0240850390351533, 0.0997405901975205, -0.122937515437584],
                 [-0.0888251825767337, -0.688740621952690, 0.719545993596271, 0.0270464000815531],
                 [0.0513651035248441, -0.724607657395577, -0.687244766444555, -0.123528165073962],
                 [0, 0, 0, 1]])
T[3] = np.array([[-0.0273488317706124, -0.188484250922263, 0.981695333876583, 0.212195561125076],
                 [0.998347172190618, 0.0445084100068802, 0.0363582895314877, -0.0328576885487003],
                 [-0.0505466633891773, 0.981067117272276, 0.186955465893052, 0.589376441177543],
                 [0, 0, 0, 1]])


def project_point_cloud(xyz_pts, feature_src, target_camid):
    """
    function to project a given feature vector to a target camera plane by using the point cloud

    :param xyz_pts:
        [B,3,W,H] tensor: point cloud points
    :param feature_src:
        [B,C+1,H,W]: features vector, including softmax scores of each class + depth
    :param target_camid:
        int: camera ID

    :return:
        [B,C+1,H,W]: flipped feature vector
    """

    # flatten feature
    batch = feature_src.shape[0]
    channel = feature_src.shape[1]
    feature_src_flat = torch.flatten(feature_src.transpose(2, 3), 2)

    # normalize depth
    xyz_pts = xyz_pts.view(batch, 3, -1)  # format it as dimension Bx3xN
    z = xyz_pts[:, 2, :].view(batch, 1, -1).expand(-1, 3, -1)
    xyz_pts_norm = (xyz_pts / z).type(torch.double)

    # project
    im_pts = torch.matmul(CAM_MTX[target_camid], xyz_pts_norm)
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
    feature_dst = feature_dst.reshape(batch, channel, HEIGHT, WIDTH)
    return torch.flip(feature_dst, [2, 3])  # needs to flip horizontally and vertically


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
    xyz_homogeneous = torch.cat((xyz_src, torch.ones(batch, 1, HEIGHT, WIDTH)), dim=1)  # shape Bx4xHxW
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
        self.normalize = T.Normalize(mean=[0.5] * num_classes, std=[0.5] * num_classes)

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
        # TODO: verify this
        feature = self.normalize(feature)
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
