import torch
import random
import numpy as np
from skimage import transform, filters
from PIL import Image, ImageOps, ImageFilter


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


class RandomVerticalFlip(object):
    def __init__(self, temporal=False):
        self.temporal = temporal

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        random_image = sample['random_image'] if self.temporal else None
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            random_image = random_image.transpose(Image.FLIP_TOP_BOTTOM) if self.temporal else None

        return {'image': img,
                'label': mask,
                'random_image': random_image}


class RandomRotate(object):
    def __init__(self, degree, temporal=False):
        self.degree = degree
        self.temporal = temporal

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        random_image = sample['random_image'] if self.temporal else None
        rotate_degree = random.uniform(-1 * self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)
        random_image = random_image.rotate(rotate_degree, Image.BILINEAR) if self.temporal else None

        return {'image': img,
                'label': mask,
                'random_image': random_image}


class RandomGaussianBlur(object):
    def __init__(self, temporal=False):
        self.temporal = temporal

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        random_image = sample['random_image'] if self.temporal else None
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
            random_image = random_image.filter(
                ImageFilter.GaussianBlur(radius=random.random())) if self.temporal else None

        return {'image': img,
                'label': mask,
                'random_image': random_image}


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


class FixScaleCrop(object):
    def __init__(self, crop_size, temporal):
        self.crop_size = crop_size
        self.temporal = temporal

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        if self.temporal:
            random_image = sample['random_image']
            random_image = random_image.resize((ow, oh), Image.BILINEAR)
            random_image = random_image.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        else:
            random_image = None

        return {'image': img,
                'label': mask,
                'random_image': random_image}


class FixedResize(object):
    def __init__(self, size, temporal):
        self.size = (size, size)  # size: (h, w)
        self.temporal = temporal

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,
                'label': mask,
                'random_image': random_image}
