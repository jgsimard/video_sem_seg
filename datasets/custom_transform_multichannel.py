import torch
import random
import numpy as np
from skimage import transform, filters
from PIL import Image, ImageOps, ImageFilter
import albumentations as aug
from albumentations.core.transforms_interface import ImageOnlyTransform


def ToTensor(sample, batch, channel):
    """Convert ndarrays  H x W x B*C" in sample to
    Tensors B x C x H x W."""

    img = torch.from_numpy(sample['image']).float().cuda()
    mask = torch.from_numpy(sample['mask']).float().cuda()

    img = img.permute(2, 0, 1)
    img = img.view(batch, channel, img.shape[1], img.shape[2])
    mask = mask.permute(2, 0, 1)

    return {'image': img,
            'label': mask}


def ToNumpy(sample):
    """Convert tensor B x C x H x W to
    ndarrays H x W x B*C"""

    img = sample['image']
    mask = sample['label']

    img = img.view(-1, img.shape[2], img.shape[3])
    img = img.permute(1, 2, 0)
    mask = mask.permute(1, 2, 0)

    img = img.cpu().numpy()
    mask = mask.cpu().numpy()

    return {'image': img,
            'label': mask}


def RandomDropOut(feature_in, CAM_NUM, p=0.5):
    # input is CAM_NUM x C x H x W

    for camid in range(1, CAM_NUM):
        if random.random() < p:
            feature_in[camid, :, :, :] = 0

    return feature_in


class ToPILImage(object):
    """Convert tensor to PIL list of images"""

    def __call__(self, sample):
        # img is tensor in B x C x H x W
        # mask is tensor in B x H x W

        img = sample["image"]
        mask = sample["label"]

        batch = img.shape[0]
        channel = img.shape[1]

        img_PIL = [None] * batch
        mask_PIL = [None] * batch
        for b in range(batch):
            mask_PIL[b] = Image.fromarray(mask[b, :, :])
            img_PIL_Channel = [None] * channel
            img_PIL[b] = img_PIL_Channel
            for c in range(channel):
                img_PIL_Channel[c] = Image.fromarray(img[b, c, :, :])

        return {'image': img_PIL,
                'label': mask_PIL}


if __name__ == "__main__":
    from torchvision import transforms
    import matplotlib.pyplot as plt
    from PIL import Image

    # load tensor
    img = Image.open(
        '/home/deepsight3/dev/deepsight/MultiView/matlab/Multi-Cam/Data/CalibrationData_2019_06_07/JPGs/MultiView_2019_06_07_11_00_46.jpg').convert(
        'L')
    img = np.array(img)
    img = img[None, None, ...]
    img = np.concatenate((img,) * 4, axis=0)
    img = np.concatenate((img,) * 13, axis=1)
    img = torch.Tensor(img)
    mask = torch.randint(0, 5, [4, 287, 352])

    # convert to numpy
    sample = ToNumpy({'image': img, 'label': mask})
    img = sample['image']
    mask = sample['label']
    plt.imshow(img[:, :, 0])
    plt.show()

    # augment
    trans = aug.Compose([aug.HorizontalFlip(p=1),
                         aug.ShiftScaleRotate(p=1, rotate_limit=20, border_mode=0),
                         aug.Blur(p=1, blur_limit=3),
                         aug.Cutout(p=1, num_holes=128, max_h_size=3, max_w_size=3)
                         ])
    sample = trans(image=img, mask=mask)
    img = sample['image']
    plt.imshow(img[:, :, 0])
    plt.show()

    # convert to tensor
    sample = ToTensor(sample, batch=4, channel=13)
    img = sample['image']
    plt.imshow(img[0, 0, :, :].cpu().numpy())

    # dropout
    img = RandomDropOut(img, 4, p=0.5)
    for camid in range(4):
        plt.imshow(img[camid, 3, :, :, ].cpu().numpy())
        plt.show()