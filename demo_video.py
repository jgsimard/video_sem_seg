import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
import sys
import signal

import models.convcrf as convcrf
from datasets import custom_transforms as tr
from models.modeling.deeplab import DeepLab
from models.modeling.sync_batchnorm.replicate import patch_replication_callback
from utils.visualization import fig2img, vis_segmentation
from utils.utils import get_files, create_directory, timeit
from train import get_args
import time
from models.afp import LowLatencyModel


def load_model(args, nclass=11, temporal=False):
    if not temporal:
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
    else:
        spatial_model = DeepLab(num_classes=nclass,
                                backbone=args.backbone,
                                output_stride=args.out_stride,
                                sync_bn=args.sync_bn,
                                freeze_bn=True)
        # Fix deeplab as the features extractor
        for param in spatial_model.parameters():
            param.requires_grad = False

        temporal_model = LowLatencyModel(spatial_model, kernel_size=args.svc_kernel_size, flow=args.flow, fixed_schedule=args.demo_frame_fixed_schedule)

        # CUDA
        spatial_model = torch.nn.DataParallel(spatial_model, device_ids=args.gpu_ids)
        patch_replication_callback(spatial_model)
        spatial_model = spatial_model.cuda()

        temporal_model = torch.nn.DataParallel(temporal_model, device_ids=args.gpu_ids)
        patch_replication_callback(temporal_model)
        temporal_model = temporal_model.cuda()

        # LOAD
        checkpoint = torch.load(args.resume)
        spatial_model.module.load_state_dict(checkpoint['spatial_model_state_dict'])
        temporal_model.module.load_state_dict(checkpoint['temporal_model_state_dict'])

        #EVAL
        temporal_model.eval()

        return temporal_model


rgb_transform = transforms.Compose([tr.FixScaleCrop(crop_size=513),
                                    tr.Normalize(mean=(0.5, 0.5, 0.5),
                                                 std=(0.5, 0.5, 0.5)),
                                    tr.ToTensor()])

class DeepSightDemoRGB(Dataset):
    NUM_CLASSES = 11
    CLASSES = ['background', 'ortable', 'psc', 'vsc', 'human', 'cielinglight', 'mayostand', 'table', 'anesthesiacart', 'cannula', 'instrument']

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
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        label = Image.open(path)
        sample = {'image': img, 'label': label}
        sample = self.transform(sample)
        sample["id"] = self.images[item]
        return sample



class VideoWriter(object):
    def __init__(self, directory, name="inference_video", fps=30, shape=(1500,500)):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer =  cv2.VideoWriter(os.path.join(directory, f'{name}_{int(time.time())}.mp4'), fourcc, fps, shape)

    def __enter__(self):
        return self.video_writer

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.video_writer.release()

def inference(image, model):
    imgs = []
    imgs_np = []
    masks = []
    ts = time.time()
    output = model(image)
    # output, flow = model(image)
    output = torch.argmax(output, dim=1)
    te = time.time()
    print(f'inference : {(te - ts) * 1000:2.2f} ms')

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
    # return imgs, imgs_np, masks, flow

def signal_handler(sig, frame, video_writer):
    print('You pressed Ctrl+C!')
    video_writer.release()
    sys.exit(0)

if __name__ == "__main__":
    args = get_args()
    model = load_model(args, nclass=11, temporal=args.demo_temporal)

    if args.demo_img_folder is not None:
        # rgb_demo_dataset = DeepSightDemoRGB(args.demo_img_folder)
        rgb_demo_dataset = DeepSightDemoDepth(args.demo_img_folder)
        data_loader = DataLoader(rgb_demo_dataset, batch_size=32, shuffle=True)
        pred_dir = os.path.join(args.demo_img_folder, "pred")
        transform_dir = os.path.join(args.demo_img_folder, "transform")
        create_directory(pred_dir)
        create_directory(transform_dir)
        for i, sample in enumerate(tqdm(data_loader)):
            image, target, names = sample['image'], sample['label'], sample['id']
            imgs, imgs_np, masks, flow = inference(image, model)
            save_image(flow, os.path.join(pred_dir, "flow.png"))
            for i in range(len(imgs)):
                masks[i].save(os.path.join(pred_dir, names[i]))
                imgs[i].save(os.path.join(transform_dir, names[i]))

        # create the video
        # images = sorted(get_files(transform_dir))
        images = sorted(get_files(transform_dir), key=lambda x: int(x.split(".")[0]))
        # images = sorted(get_files(transform_dir), key=lambda x: int(x.split(".")[0][14:]))
        print(images)
        fig = None
        with VideoWriter(pred_dir, name="test_imgs", fps=20) as video_writer:
            for img_name in images:
                img = np.array(Image.open(os.path.join(transform_dir, img_name)))
                pred = np.array(Image.open(os.path.join(pred_dir, img_name)))

                fig = vis_segmentation(img, pred, fig)
                data = fig2img(fig)
                video_writer.write(data)


    stream = None
    if args.demo_video_path is not None:
        stream = cv2.VideoCapture(args.demo_video_path)
    elif args.demo_camera:
        stream = cv2.VideoCapture(0)


    if stream is not None:
        fig = None
        output_video_directory = args.demo_video_output if args.demo_video_output is not None else os.getcwd()
        output_video_directory = args.demo_video_output if args.demo_video_output is not None else os.getcwd()

        with VideoWriter(output_video_directory, name="test_video_spatial") as video_writer:
            signal.signal(signal.SIGINT, lambda x,y: signal_handler(x,y,video_writer))
            try:
                while(stream.isOpened()):
                    read_succesful, frame = stream.read()
                    if not read_succesful:
                        print(read_succesful)
                        continue

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    #for ZED
                    # frame_w = frame.shape[1]
                    # frame = frame[:,:int(frame_w/2), :]
                    # frame = cv2.flip(frame, 0)
                    print(frame.shape)
                    img = Image.fromarray(frame)
                    sample = {'image':img, 'label':img}
                    sample = rgb_transform(sample)

                    image = sample['image'].unsqueeze(dim=0)

                    imgs, imgs_np, masks = inference(image, model)
                    # imgs, imgs_np,  masks, flow = inference(image, model)
                    # if flow is not None:
                    #     print("flow")
                    #     save_image(flow, os.path.join("/home/deepsight2/jg_internship/video_sem_seg", "flow.png"))
                    classes=['background', 'ortable', 'psc', 'vsc', 'human', 'cielinglight', 'mayostand', 'table', 'anesthesiacart', 'cannula', 'instrument']
                    fig = vis_segmentation(imgs_np[0], masks[0], fig, classes=classes)
                    data = fig2img(fig)
                    video_writer.write(data)
            except KeyboardInterrupt:
                print("KeyboardInterrupt : Bye bye bye")
                sys.exit()



