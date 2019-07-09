import os
from torch.utils.data import Dataset, DataLoader
from os.path import join
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from datasets import custom_transforms as tr


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

        self.transform_validation = transforms.Compose([tr.FixScaleCrop(crop_size=1080),
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

if __name__ == "__main__":
    root_dir = "/home/deepsight2/development/data/rgb"
    set = "validation"
    rgb_dataset = DeepSightRGB(root_dir, set)
    fig = plt.figure()
    print(len(rgb_dataset))
    for i in range(len(rgb_dataset)):
        sample = rgb_dataset[i]

        print(i, sample['image'].shape, sample['label'].shape)

        plt.imshow(np.transpose(np.asarray(sample['image']), (1, 2, 0)))
        plt.show()
        plt.imshow(sample['label'])
        plt.show()

        if i == 5:
            break

    rgb_dataset = DeepSightRGB(root_dir, set)

    dataloader = DataLoader(rgb_dataset,
                            batch_size=4,
                            shuffle=True,
                            num_workers=4)

    for i_batch, sample_batched in enumerate(dataloader):
        print(sample_batched['image'].shape)
        break
        # print(i_batch, sample_batched['image'].size(), sample_batched['mask'].size())


