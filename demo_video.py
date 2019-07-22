import argparse
import os
import numpy as np
from tqdm import tqdm

import torch

from mypath import Path
from datasets import make_data_loader
from models.modeling.sync_batchnorm.replicate import patch_replication_callback
from models.modeling.deeplab import DeepLab
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import custom_transforms as tr
from PIL import Image
from torchvision.utils import save_image
from matplotlib import gridspec
from matplotlib import pyplot as plt
import cv2


import models.convcrf as convcrf

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

    # crf
    crf = None
    if args.GaussCrf:
        if args.TrainCrf:
            conf = convcrf.isi_trainable_conf
        else:
            conf = convcrf.isi_untrainable_conf

        if args.dataset == "isi_rgb":
            shape = (513, 513)
        elif args.dataset == "isi_intensity":
            shape = (287, 352)
        crf = convcrf.GaussCRF(conf=conf, shape=shape, nclasses=nclass)

        #crf on cuda
        crf = torch.nn.DataParallel(crf, device_ids=args.gpu_ids)
        patch_replication_callback(crf)
        crf = crf.cuda()
        if 'crf_state_dict' in checkpoint:
            if checkpoint['crf_state_dict'] is not None:
                print("loading crf")
                crf.module.load_state_dict(checkpoint['crf_state_dict'])

    return model, crf

def get_args():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus DEMO")
    parser.add_argument('--backbone',
                        type=str,
                        default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride',
                        type=int,
                        default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset',
                        type=str,
                        default='isi',
                        choices=['pascal', 'coco', 'cityscapes', 'isi_rgb', 'isi_intensity', 'isi_depth', 'isi_intensi'],
                        help='dataset name (default: isi)')
    parser.add_argument('--workers',
                        type=int,
                        default=4,
                        metavar='N',
                        help='dataloader threads')
    parser.add_argument('--base-size',
                        type=int,
                        default=513,
                        help='base image size')
    parser.add_argument('--crop-size',
                        type=int,
                        default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn',
                        type=bool,
                        default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn',
                        type=bool,
                        default=False,
                        help='whether to freeze bn parameters (default: False)')

    # cuda, seed and logging
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu-ids',
                        type=str,
                        default='0',
                        help='use which gpu to train, must be a comma-separated list of integers only (default=0)')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume',
                        type=str,
                        default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname',
                        type=str,
                        default=None,
                        help='set the checkpoint name')
    # ISI
    parser.add_argument('--dataset_dir',
                        type=str,
                        default='/home/deepsight2/development/data/rgb',
                        help='put the path to dataset root dir')
    parser.add_argument('--GaussCrf',
                        action='store_true',
                        default=False,
                        help='Add GaussCRF at the end of the model')
    parser.add_argument('--TrainCrf',
                        action='store_true',
                        default=False,
                        help='Train GaussCRF at the end of the model')
    parser.add_argument('--crf_start_epoch',
                        type=int,
                        default=0,
                        help='Epoch at which to start training the CRF (not stable if trained from the begining!)')
    parser.add_argument('--imgs_folder',
                        type=str,
                        default=None,
                        help='folder containing images to be processed')
    parser.add_argument('--video_path',
                        type=str,
                        default=None,
                        help='path to a video to be processed')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    if args.checkname is None:
        args.checkname = 'deeplab-' + str(args.backbone)
    return args

class DeepSightDemoRGB(Dataset):
    NUM_CLASSES = 11
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = transforms.Compose([tr.FixScaleCrop(crop_size=513),
                                            tr.Normalize(mean=(0.5, 0.5, 0.5),
                                                         std=(0.5, 0.5, 0.5)),
                                            tr.ToTensor()])

        self.images = [f for f in  os.listdir(self.root_dir) if os.path.isfile(os.path.join(self.root_dir, f))]
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

        self.images = [f for f in  os.listdir(self.root_dir) if os.path.isfile(os.path.join(self.root_dir, f))]
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

# def main():
#     args = get_args()
#     if args.imgs_folder:
#         dataset =
#     print(args)
#     model, crf = load_model(args)
#     for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
#         trainer.training(epoch)
#         if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
#             trainer.validation(epoch)
#
#     trainer.writer.close()

def get_files(path):
    return [f for f in  os.listdir(path) if os.path.isfile(os.path.join(path, f))]

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A colormap for visualizing segmentation results.
  """
  # Max number of entries in the colormap for each dataset.
  # Dataset names.
  _ADE20K = 'ade20k'
  _CITYSCAPES = 'cityscapes'
  _PASCAL = 'pascal'

  _DATASET_MAX_ENTRIES = {
      _ADE20K: 151,
      _CITYSCAPES: 19,
      _PASCAL: 256,
  }

  def bit_get(val, idx):
      """Gets the bit value.

      Args:
        val: Input value, int or numpy int array.
        idx: Which bit of the input val.

      Returns:
        The "idx"-th bit of input val.
      """
      return (val >> idx) & 1

  colormap = np.zeros((_DATASET_MAX_ENTRIES[_PASCAL], 3), dtype=int)
  ind = np.arange(_DATASET_MAX_ENTRIES[_PASCAL], dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= bit_get(ind, channel) << shift
    ind >>= 3

  return colormap

def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
      label: A 2D array with integer type, storing the segmentation label.

    Returns:
      result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.

    Raises:
      ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]

def vis_segmentation(image, seg_map, fig = None):
    """Visualizes input image, segmentation map and overlay view."""
    # plt.ion()
    if fig is None:
        fig = plt.figure(figsize=(15, 5))
    seg_image, overlay_image = label2rgb(seg_map, image)
    grid_spec = gridspec.GridSpec(1, 3, width_ratios=[6, 6, 6])

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('input image')

    plt.subplot(grid_spec[1])
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('segmentation map')

    plt.subplot(grid_spec[2])
    plt.imshow(overlay_image)
    plt.axis('off')
    plt.title('segmentation overlay')

    # unique_labels = np.unique(seg_map)
    # ax = plt.subplot(grid_spec[3])
    # plt.imshow(FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
    # ax.yaxis.tick_right()
    # plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
    # plt.xticks([], [])
    # ax.tick_params(width=0.0)
    # plt.grid('off')
    # fig.canvas.draw()
    # plt.close()
    # plt.ion()
    # plt.show()
    # plt.pause(0.01)
    plt.ion()
    # plt.imshow(data)
    plt.show()
    plt.pause(0.01)
    return fig

def fig2img(fig):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)  # COLOR_BGR2RGB
    return data

def label2rgb(lbl, img=None, n_labels=None, alpha=0.5, colors=None):
    mask = create_pascal_label_colormap()[lbl].astype(np.uint8)
    # mask = np.expand_dims(lbl, axis=2).astype(np.uint8)

    if img is not None:
        img_gray = Image.fromarray(img).convert('LA')
        img_gray = np.asarray(img_gray.convert('RGB'))
        overlay = alpha * mask + (1 - alpha) * img_gray
        overlay = overlay.astype(np.uint8)
    else:
        overlay = mask

    return mask, overlay

if __name__ == "__main__":
    args = get_args()
    # main()
    # path = "/home/deepsight/DeepSightData/Example ToF sequences for inferrence/LabelMeImages/Docked 1"
    # depth_demo_dataset = DeepSightDemoDepth(path)
    model, crf = load_model(args, nclass=11)
    model.eval()
    crf.eval()

    for id in ['33', '46', '80', '18']:
        path = f"/home/deepsight/data/rgb/validation_video/{id}"
        depth_demo_dataset = DeepSightDemoRGB(path)

        print(len(depth_demo_dataset))
        data_loader = DataLoader(depth_demo_dataset, batch_size=16, shuffle=True)
        pred_dir = os.path.join(path, "pred")
        transform_dir = os.path.join(path, "transform")
        create_directory(pred_dir)
        create_directory(transform_dir)


        for i, sample in enumerate(tqdm(data_loader)):
            image, target, names = sample['image'], sample['label'], sample['id']
            output = model(image)
            if crf is not None:
                output = crf(output, image)
            output = torch.argmax(output, dim=1)
            for i in range(output.shape[0]):
                img = Image.fromarray(output[i,:,:].cpu().numpy().astype(np.uint8))
                img.save(os.path.join(pred_dir,names[i]))

                img = Image.fromarray(np.transpose(((image[i,:, :, :] * 0.5 + 0.5)*255).cpu().numpy().astype(np.uint8), (1,2,0)))
                img.save(os.path.join(transform_dir, names[i]))

            # output = crf(model(image), image)
            # print(output.shape, names)

        images = sorted(get_files(transform_dir),key=lambda x : int(x.split(".")[0]))
        print(images)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f'{pred_dir}_video.mp4', fourcc, 10.0, (1500, 500))

        fig = None
        for img_name in images:
            img = np.array(Image.open(os.path.join(transform_dir, img_name)))
            pred = np.array(Image.open(os.path.join(pred_dir, img_name)))

            fig = vis_segmentation(img, pred, fig)
            data = fig2img(fig)
            out.write(data)
        out.release()



