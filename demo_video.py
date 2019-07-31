import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

import models.convcrf as convcrf
from datasets import custom_transforms as tr
from models.modeling.deeplab import DeepLab
from models.modeling.sync_batchnorm.replicate import patch_replication_callback
from utils.visualization import fig2img, vis_segmentation
from utils.utils import get_files, create_directory
from train import get_args


def load_model(args, nclass=11):
    # loading saved model
    if not os.path.isfile(args.resume):
        raise RuntimeError(f"=> no checkpoint found at '{args.resume}'")
    checkpoint = torch.load(args.resume)

    # deeplab
    model = DeepLab(num_classes=nclass,
                    backbone=args.backbone,
                    output_stride=args.out_stride,
                    sync_bn=args.sync_bn,
                    freeze_bn=args.freeze_bn)

    model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
    patch_replication_callback(model)
    model = model.cuda()

    model.module.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

rgb_transform = transforms.Compose([tr.FixScaleCrop(crop_size=513),
                                    tr.Normalize(mean=(0.5, 0.5, 0.5),
                                                 std=(0.5, 0.5, 0.5)),
                                    tr.ToTensor()])

class DeepSightDemoRGB(Dataset):
    NUM_CLASSES = 11

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = rgb_transform
        self.images = [f for f in os.listdir(self.root_dir) if os.path.isfile(os.path.join(self.root_dir, f))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        path = os.path.join(self.root_dir, self.images[item])
        img = Image.open(path)
        label = Image.open(path)
        sample = {'image': img, 'label': label}
        sample = self.transform(sample)
        sample["id"] = self.images[item]
        return sample


class DeepSightDemoDepth(Dataset):
    NUM_CLASSES = 13

    def __init__(self, root_dir, split="train"):
        self.root_dir = root_dir
        self.transform = transforms.Compose([tr.Normalize(mean=(0.5, 0.5, 0.5),
                                                          std=(0.5, 0.5, 0.5)),
                                             tr.ToTensor()])

        self.images = [f for f in os.listdir(self.root_dir) if os.path.isfile(os.path.join(self.root_dir, f))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        path = os.path.join(self.root_dir, self.images[item])
        img = Image.open(path)
        label = Image.open(path)
        sample = {'image': img, 'label': label}
        sample = self.transform(sample)
        sample["id"] = self.images[item]
        return sample


# def get_video_writer(directory, name="inference_video", fps=30, shape=(1500,500)):
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(os.path.join(directory, f'{name}.mp4'), fourcc, fps, shape)
#     return out


class VideoWriter(object):
    def __init__(self, directory, name="inference_video", fps=30, shape=(1500,500)):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer =  cv2.VideoWriter(os.path.join(directory, f'{name}.mp4'), fourcc, fps, shape)

    def __enter__(self):
        return self.video_writer

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.video_writer.release()

def inference(image, model):
    imgs = []
    imgs_np = []
    masks = []
    output = model(image)
    print(output.shape)
    output = torch.argmax(output, dim=1)
    print(output.shape)
    for i in range(output.shape[0]):
        mask = output[i, :, :].cpu().numpy().astype(np.uint8)
        mask = Image.fromarray(mask)
        masks.append(mask)

        img = (image[i, :, :, :] * 0.5 + 0.5) * 255
        img = img.cpu().numpy().astype(np.uint8)
        img = np.transpose(img, (1, 2, 0))
        imgs_np.append(img)

        img = Image.fromarray(img)
        imgs.append(img)

    return imgs, imgs_np, masks


if __name__ == "__main__":
    args = get_args()
    model = load_model(args)
    if args.demo_video_path is not None:
        stream = cv2.VideoCapture(args.demo_video_path)
    elif args.demo_camera:
        stream = cv2.VideoCapture(1)
    fig = None
    output_video_directory = args.demo_video_output if args.demo_video_output is not None else os.getcwd()
    with VideoWriter(output_video_directory, name="test_camera") as video_writer:
        while(stream.isOpened()):
            read_succesful, frame = stream.read()
            if not read_succesful:
                print(read_succesful)
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # frame = cv2.flip(frame, 0)
            img = Image.fromarray(frame)
            sample = {'image':img, 'label':img}
            sample = rgb_transform(sample)

            image = sample['image'].unsqueeze(dim=0)

            print(image.shape)
            imgs, imgs_np,  masks = inference(image, model)
            print(imgs, masks)

            fig = vis_segmentation(imgs_np[0], masks[0], fig)
            data = fig2img(fig)
            video_writer.write(data)



    # for id in ['33', '46', '80', '18']:
    #     path = f"/home/deepsight/data/rgb/validation_video/{id}"
    #     depth_demo_dataset = DeepSightDemoRGB(path)
    #
    #     print(len(depth_demo_dataset))
    #     data_loader = DataLoader(depth_demo_dataset, batch_size=16, shuffle=True)
    #     pred_dir = os.path.join(path, "pred")
    #     transform_dir = os.path.join(path, "transform")
    #     create_directory(pred_dir)
    #     create_directory(transform_dir)
    #
    #     for i, sample in enumerate(tqdm(data_loader)):
    #         image, target, names = sample['image'], sample['label'], sample['id']
    #         output = model(image)
    #         output = torch.argmax(output, dim=1)
    #         for i in range(output.shape[0]):
    #             img = Image.fromarray(output[i, :, :].cpu().numpy().astype(np.uint8))
    #             img.save(os.path.join(pred_dir, names[i]))
    #
    #             img = Image.fromarray(
    #                 np.transpose(((image[i, :, :, :] * 0.5 + 0.5) * 255).cpu().numpy().astype(np.uint8), (1, 2, 0)))
    #             img.save(os.path.join(transform_dir, names[i]))
    #
    #
    #     images = sorted(get_files(transform_dir), key=lambda x: int(x.split(".")[0]))
    #     print(images)
    #
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #     out = cv2.VideoWriter(f'{pred_dir}_video.mp4', fourcc, 10.0, (1500, 500))
    #
    #     fig = None
    #     for img_name in images:
    #         img = np.array(Image.open(os.path.join(transform_dir, img_name)))
    #         pred = np.array(Image.open(os.path.join(pred_dir, img_name)))
    #
    #         fig = vis_segmentation(img, pred, fig)
    #         data = fig2img(fig)
    #         out.write(data)
    #     out.release()
