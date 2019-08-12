import torch
import random
import numpy as np
from skimage import transform, filters
from PIL import Image, ImageOps, ImageFilter
import albumentations as aug
from albumentations.core.transforms_interface import ImageOnlyTransform, to_tuple


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


class RandomNoise(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5, gauss_mean=0.0, gauss_var=0.01, sp_ratio=0.5, sp_amount=0.005):
        super(RandomNoise, self).__init__(always_apply, p)
        self.gauss_mean = gauss_mean
        self.gauss_var = gauss_var
        self.sp_ratio = sp_ratio
        self.sp_amount = sp_amount

    def apply(self, image, **params):
        prob = random.random()
        if 0.0 <= prob < 0.5:
            # print("gauss")
            image = self.noise("gauss", image)
        else:
            # print("s&p")
            image = self.noise("s&p", image)
        # else:
        #     # print("poisson")
        #     image = self.noise("poisson", image)

        return image

    def noise(self, noise_type, image):
        row, col, ch = image.shape
        if noise_type == "gauss":
            mean = self.gauss_mean
            sigma = self.gauss_var ** 0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            noisy = image + gauss
        elif noise_type == "s&p":
            s_vs_p = self.sp_ratio
            amount = self.sp_amount
            noisy = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = tuple([np.random.randint(0, i - 1, int(num_salt))
                            for i in image.shape])
            noisy[coords] = 1

            # Pepper mode
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = tuple([np.random.randint(0, i - 1, int(num_pepper))
                            for i in image.shape])
            noisy[coords] = 0
        # elif noise_type == "poisson":
        #     vals = len(np.unique(image))
        #     vals = 2 ** np.ceil(np.log2(vals))
        #     noisy = image + np.random.poisson(image * vals) / float(vals)

        return noisy


class RandomDepthShift(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5, limit=2.0, n_class=13, n_camera=4):
        super(RandomDepthShift, self).__init__(always_apply, p)
        self.limit = to_tuple(limit)
        self.n_class = n_class
        self.n_camera = n_camera

    def apply(self, img, shift=0., **params):
        ind = torch.empty([self.n_camera])
        for i in range(self.n_camera):
            ind[i] = self.n_camera * (self.n_class + 1) - 1
        ind = ind.long()
        img[:, :, ind] = img[:, :, ind] + shift
        return img

    def get_params(self):
        return {
            'shift': random.uniform(self.limit[0], self.limit[1])
        }


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
    trans = aug.Compose([aug.HorizontalFlip(p=0.0),
                         aug.ShiftScaleRotate(p=0.0, rotate_limit=20, border_mode=0),
                         aug.Blur(p=0.0, blur_limit=3),
                         RandomNoise(p=1),
                         RandomDepthShift(p=1)
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
