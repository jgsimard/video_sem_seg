import torch
import random
import numpy as np
from skimage import transform, filters
from PIL import Image, ImageOps, ImageFilter
from albumentations.pytorch import *
from albumentations import *


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.), temporal=False, multiview=False):
        self.mean = mean
        self.std = std
        self.temporal = temporal
        self.multiview = multiview

    def __call__(self, sample):
        # input is numpy array in CAM_NUM x C x H x W
        img = sample['image']
        mask = sample['label']

        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)

        img /= 255.0
        img -= self.mean
        img /= self.std

        if self.temporal:
            random_image = np.array(sample['random_image']).astype(np.float32)
            random_image /= 255.0
            random_image -= self.mean
            random_image /= self.std

            return {'image': img,
                    'label': mask,
                    'random_image': random_image}

        elif self.multiview:
            pointcloud = sample['pointcloud']
            return {'image': img,
                    'label': mask,
                    'pointcloud': pointcloud}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, temporal=False, multiview=False):
        self.temporal = temporal
        self.multiview = multiview

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        if self.temporal:
            random_image = np.array(sample['random_image']).astype(np.float32).transpose((2, 0, 1))
            random_image = torch.from_numpy(random_image)
            return {'image': img,
                    'label': mask,
                    'random_image': random_image}
        elif self.multiview:
            pointcloud = sample['pointcloud']
            pointcloud = np.array(pointcloud).astype(np.float32).transpose((0, 2, 1))
            pointcloud = torch.from_numpy(pointcloud).float()
            return {'image': img,
                    'label': mask,
                    'pointcloud': pointcloud}


# class ToTensor(object):
#     """Convert ndarrays in sample to Tensors."""
#
#     def __call__(self, sample):
#         sample['image'] = torch.from_numpy(sample['image']).float()
#         sample['depth'] = torch.from_numpy(sample['depth']).float()
#         sample['label'] = torch.from_numpy(sample['label']).float()
#         sample['pointcloud'] = torch.from_numpy(sample['pointcloud']).float()
#
#         return sample


# class ToNumpy(object):
#     """Convert tensor to ndarrays"""
#
#     def __call__(self, sample):
#         sample['image'] = sample['image'].cpu().numpy()
#         sample['depth'] = sample['depth'].cpu().numpy()
#         sample['label'] = sample['label'].cpu().numpy()
#         sample['pointcloud'] = sample['pointcloud'].cpu().numpy()
#
#         return sample


# class ToPILImage(object):
#     """Convert tensor to PIL list of images"""
#
#     def __call__(self, sample):
#         # img is tensor in B x C x H x W
#         # mask is tensor in B x H x W
#
#         img = sample["image"]
#         mask = sample["label"]
#
#         batch = img.shape[0]
#         channel = img.shape[1]
#
#         img_PIL = [None] * batch
#         mask_PIL = [None] * batch
#         for b in batch:
#             mask_PIL = Image.fromarray(mask[b, :, :].numpy)
#             img_PIL = [None] * channel
#             for c in channel:
#                 img_PIL = Image.fromarray(img[b, c, :, :].numpy)
#
#         return {'image': img_PIL,
#                 'label': mask_PIL}


class RandomHorizontalFlip(object):
    def __init__(self, temporal=False):
        self.temporal = temporal

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        random_image = sample['random_image'] if self.temporal else None
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            random_image = random_image.transpose(Image.FLIP_LEFT_RIGHT) if self.temporal else None
        return {'image': img,
                'label': mask,
                'random_image': random_image}


class RandomHorizontalFlipMultiView(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        # input is tensor in B x C x H x W
        img = sample['image']
        label = sample['label']

        if random.random() < self.p:
            img = torch.flip(img, dims=[-1])
            label = torch.flip(label, dims=[-1])

        return {'image': img,
                'label': label}


class RandomRotate(object):
    def __init__(self, degree, p=0.5):
        self.degree = degree
        self.p = p

    def __call__(self, sample):
        # input is tensor in B x C x H x W
        img = sample['image'].cpu().numpy()
        mask = sample['label'].cpu().numpy()
        batch = img.shape[0]
        channel = img.shape[1]
        rotate_degree = random.uniform(-1 * self.degree, self.degree)

        if random.random() < self.p:
            for b in range(batch):
                mask[b, :, :] = transform.rotate(mask[b, :, :], rotate_degree, mode='constant', preserve_range=True)

                for c in range(channel):
                    img[b, c, :, :] = transform.rotate(img[b, c, :, :], rotate_degree, mode='constant')

        return {'image': torch.tensor(img).cuda(),
                'label': torch.tensor(mask).cuda()}


class RandomGaussianBlur(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        img = sample['image'].cpu().numpy()
        mask = sample['label'].cpu().numpy()
        batch = img.shape[0]
        channel = img.shape[1]

        if random.random() < self.p:
            for b in range(batch):
                for c in range(channel):
                    img[b, c, :, :] = filters.gaussian(img[b, c, :, :], sigma=random.random())

        return {'image': torch.tensor(img).cuda(),
                'label': torch.tensor(mask).cuda()}


class RandomDropOut(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, feature_in, CAM_NUM):
        # input is B x CAM_NUM x C x H x W

        for camid in range(1, CAM_NUM):
            if random.random() < self.p:
                feature_in[:, camid, :, :, :] = 0

        return feature_in


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0, scales=(0.5, 1.5), temporal=False):
        if isinstance(base_size, int):
            base_size = (base_size, base_size)
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill
        self.scales = (0.5, 1.5)
        self.temporal = temporal

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        short_edge = min(self.base_size)
        short_size = random.randint(int(short_edge * self.scales[0]), int(short_edge * self.scales[1]))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size[0]:
            padh = self.crop_size[0] - oh if oh < self.crop_size[0] else 0
            padw = self.crop_size[1] - ow if ow < self.crop_size[1] else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size[1])
        y1 = random.randint(0, h - self.crop_size[0])
        img = img.crop((x1, y1, x1 + self.crop_size[1], y1 + self.crop_size[0]))
        mask = mask.crop((x1, y1, x1 + self.crop_size[1], y1 + self.crop_size[0]))

        if self.temporal:
            random_image = sample['random_image']
            random_image = random_image.resize((ow, oh), Image.BILINEAR)
            if short_size < self.crop_size[0]:
                random_image = ImageOps.expand(random_image, border=(0, 0, padw, padh), fill=0)
            random_image = random_image.crop((x1, y1, x1 + self.crop_size[1], y1 + self.crop_size[0]))
        else:
            random_image = None

        return {'image': img,
                'label': mask,
                'random_image': random_image}
