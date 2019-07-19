import argparse
import os
import numpy as np
from tqdm import tqdm

import torch

from mypath import Path
from datasets import make_data_loader
from models.modeling.sync_batchnorm.replicate import patch_replication_callback
from models.modeling.deeplab_multiview import DeepLabMultiView
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator

import models.convcrf as convcrf

import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from datasets.utils import decode_seg_map_sequence

CAM_NUM = 4


def onehot(targets, num_classes):
    """Origin: https://github.com/moskomule/mixup.pytorch
    convert index tensor into onehot tensor
    :param targets: index tensor
    :param num_classes: number of classes
    """
    # assert isinstance(targets, torch.LongTensor)
    return torch.zeros(targets.size()[0], num_classes).scatter_(1, targets.view(-1, 1), 1)


def mixup(inputs, targets, num_classes, alpha=0.4):
    """Mixup on 1x32x32 mel-spectrograms.
    """
    s = inputs.size()[0]
    weight = torch.Tensor(np.random.beta(alpha, alpha, s))
    index = np.random.permutation(s)
    x1, x2 = inputs, inputs[index, :, :, :]
    y1, y2 = onehot(targets, num_classes), onehot(targets[index,], num_classes)
    weight = weight.view(s, 1, 1, 1)
    inputs = weight * x1 + (1 - weight) * x2
    weight = weight.view(s, 1)
    targets = weight * y1 + (1 - weight) * y2
    return inputs, targets


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # Define network
        model = DeepLabMultiView(num_classes=self.nclass,
                                 backbone=args.backbone,
                                 output_stride=args.out_stride,
                                 sync_bn=args.sync_bn,
                                 freeze_bn=args.freeze_bn)

        self.crf = None
        if args.GaussCrf:
            if self.args.TrainCrf:
                conf = convcrf.isi_trainable_conf
            else:
                conf = convcrf.isi_untrainable_conf

            self.crf = convcrf.GaussCRF(conf=conf, shape=(513, 513), nclasses=self.nclass)

        # load pretrain model
        if args.path_pretrained_model is not None:
            # model.load_state_dict(torch.load(args.path_pretrained_model)['state_dict'])

            pretrained_dict = torch.load(args.path_pretrained_model)['state_dict']
            model_dict = model.deeplab.state_dict()
            # filter out uncessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            print(len(pretrained_dict))

            # overwrite existing entries
            model_dict.update(pretrained_dict)
            # load the new state dict
            model.deeplab.load_state_dict(model_dict)
            # fix parameters
            for param in model.deeplab.parameters():
                param.requires_grad = False
        else:
            # define different learning rate
            train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                            {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]
            # Define Optimizer
            optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                        weight_decay=args.weight_decay, nesterov=args.nesterov)

        # Define Optimizer
        if False:
            optimizer = torch.optim.SGD(model.merger.parameters(), lr=args.lr, momentum=args.momentum,
                                        weight_decay=args.weight_decay, nesterov=args.nesterov)
        else:
            optimizer = torch.optim.Adam(model.merger.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # # check if layers are fixed
        # for name, param in model.named_parameters():
        #     print(name, param.requires_grad)

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset + '_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=torch.ones([self.nclass, 1]).cuda(), cuda=args.cuda).build_loss(
            mode=args.loss_type)
        self.model, self.optimizer = model, optimizer

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        self.evaluator_crf = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler,
                                      args.lr,
                                      args.epochs,
                                      len(self.train_loader))

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

            if args.GaussCrf:
                self.crf = torch.nn.DataParallel(self.crf, device_ids=self.args.gpu_ids)
                patch_replication_callback(self.crf)
                self.crf = self.crf.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            print("loading checkpoint ...")
            if not os.path.isfile(args.resume):
                raise RuntimeError(f"=> no checkpoint found at '{args.resume}'")
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
                if args.GaussCrf and 'crf_state_dict' in checkpoint:
                    if checkpoint['crf_state_dict'] is not None:
                        print("loading crf")
                        self.crf.module.load_state_dict(checkpoint['crf_state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
                if args.GaussCrf and 'crf_state_dict' in checkpoint:
                    if checkpoint['crf_state_dict'] is not None:
                        print("loading crf")
                        self.crf.load_state_dict(checkpoint['crf_state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        if self.args.TrainCrf:
            self.crf.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)

        for i, sample in enumerate(tbar):
            image, target, pc = sample['image'], sample['label'], sample['pointcloud']
            # print(image.shape[0])
            # if image.shape[0] == 1:  # will fail otherwise because cannot have a batch size of 1
            #     continue

            if self.args.cuda:
                image, target, pc = image.cuda(), target.cuda(), pc.cuda()

            if self.args.loss_type == 'cem':
                image, target = mixup(image, onehot(target, 11), 11, alpha=0.4)

            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()

            # go through network
            output_single_view, output_merger = self.model(image, pc)
            if self.crf is not None and self.args.TrainCrf:
                output_merger = self.crf(output_merger, image)

            # show image
            # grid_image = make_grid(image[0, :, :, :, :][:4].clone().cpu().data, 4, normalize=True)
            # plt.imshow(np.transpose(grid_image, (1, 2, 0)))
            # plt.show()
            # hard_label = torch.max(output[0, :, :, :, :][:4], 1)[1]
            # print(hard_label.max())
            # print(hard_label.min())
            # grid_image = make_grid(
            #     decode_seg_map_sequence(hard_label.clone().cpu().numpy(),
            #                             dataset='isi'), 4, normalize=False, range=(0, 255))
            # plt.imshow(np.transpose(grid_image, (1, 2, 0)))
            # plt.show()
            # grid_image = make_grid(
            #     decode_seg_map_sequence(torch.squeeze(target[0, :, :, :][:4], 1).clone().cpu().numpy(),
            #                             dataset='isi'), 4, normalize=False, range=(0, 255))
            # plt.imshow(np.transpose(grid_image, (1, 2, 0)))
            # plt.show()

            # calculate loss
            loss = 0.0
            for camid in range(CAM_NUM):
                loss = loss + self.criterion(output_merger[:, camid, :, :, :], target[:, camid, :, :])
            loss.backward()

            # # print the gradient
            # for name, param in self.model.module.merger.named_parameters():
            #     print(name, param.grad)

            # backprop
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # Show 10 * 3 inference results each epoch
            if i % (num_img_tr // 2) == 0:
                global_step = i + num_img_tr * epoch
                self.summary.visualize_image(self.writer, self.args.dataset, image[0, :, :, :, :],
                                             target[0, :, :, :], output_single_view[0, :, :, :, :],
                                             output_merger[0, :, :, :, :],
                                             global_step)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
                'crf_state_dict': self.crf.module.state_dict() if self.crf is not None else None,
            }, is_best)

    def validation(self, epoch):
        self.model.eval()
        # self.crf.eval()
        self.evaluator.reset()
        if self.crf is not None:
            self.crf.eval()
            self.evaluator_crf.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target, pc = sample['image'], sample['label'], sample['pointcloud']
            if self.args.cuda:
                image, target, pc = image.cuda(), target.cuda(), pc.cuda()

            with torch.no_grad():
                output_single_view, output_merger = self.model(image, pc)
                if self.crf is not None:
                    # self.crf.module.CRF.npixels = (1080, 1080)
                    # self.crf.module.CRF.height = 1080
                    # self.crf.module.CRF.width = 1080
                    # print(f"output, {output.shape}, img:{image.shape}")
                    output_merger = self.crf.forward(unary=output_merger, img=image)

                # calculate loss
                loss = 0.0
                for camid in range(CAM_NUM):
                    loss = loss + self.criterion(output_merger[:, camid, :, :, :], target[:, camid, :, :])

            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))

            for camid in range(CAM_NUM):
                pred_tmp = output_merger[:, camid, :, :, :].data.cpu().numpy()
                target_tmp = target[:, camid, :, :].cpu().numpy()
                pred_tmp = np.argmax(pred_tmp, axis=1)
                # Add batch sample into evaluator
                self.evaluator.add_batch(target_tmp, pred_tmp)

            if self.crf is not None:
                pred_crf = np.argmax(output_merger.data.cpu().numpy(), axis=1)
                self.evaluator_crf.add_batch(target, pred_crf)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        # IoU = self.evaluator.Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        # for i_class, class_IoU in enumerate(IoU):
        #     print("{} class, IoU is {:1.2f}\n".format(i_class, class_IoU))
        print('Loss: %.3f' % test_loss)

        if self.crf is not None:
            # Fast test during the training
            Acc_crf = self.evaluator_crf.Pixel_Accuracy()
            Acc_class_crf = self.evaluator_crf.Pixel_Accuracy_Class()
            mIoU_crf = self.evaluator_crf.Mean_Intersection_over_Union()
            FWIoU_crf = self.evaluator_crf.Frequency_Weighted_Intersection_over_Union()
            self.writer.add_scalar('val/mIoU_crf', mIoU_crf, epoch)
            self.writer.add_scalar('val/Acc_crf', Acc_crf, epoch)
            self.writer.add_scalar('val/Acc_class_crf', Acc_class_crf, epoch)
            self.writer.add_scalar('val/fwIoU_crf', FWIoU_crf, epoch)
            print('Validation_crf:')
            print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc_crf, Acc_class_crf, mIoU_crf, FWIoU_crf))

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
                'crf_state_dict': self.crf.module.state_dict() if self.crf is not None else None,
            }, is_best)


#  pred = gausscrf.forward(unary=unary_var, img=img_var)
def get_args():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
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
                        default='isi_intensity',
                        choices=['pascal', 'coco', 'cityscapes', 'isi_intensity', 'isi_multiview'],
                        help='dataset name (default: isi)')
    parser.add_argument('--use-sbd',
                        action='store_true',
                        default=True,
                        help='whether to use SBD dataset (default: True)')
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
    parser.add_argument('--loss-type',
                        type=str,
                        default='ce',
                        choices=['ce', 'focal', 'cem'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs',
                        type=int,
                        default=None,
                        metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch',
                        type=int, default=0,
                        metavar='N',
                        help='start epochs (default:0)')
    parser.add_argument('--batch-size',
                        type=int,
                        default=None,
                        metavar='N',
                        help='input batch size for training (default: auto)')
    parser.add_argument('--test-batch-size',
                        type=int,
                        default=None,
                        metavar='N',
                        help='input batch size for testing (default: auto)')
    parser.add_argument('--use-balanced-weights',
                        action='store_true',
                        default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr',
                        type=float,
                        default=None,
                        metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler',
                        type=str,
                        default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum',
                        type=float,
                        default=0.9,
                        metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay',
                        type=float,
                        default=5e-4,
                        metavar='M',
                        help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov',
                        action='store_true',
                        default=False,
                        help='whether use nesterov (default: False)')
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
    # finetuning pre-trained models
    parser.add_argument('--ft',
                        action='store_true',
                        default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval',
                        type=int,
                        default=1,
                        help='evaluation interval (default: 1)')
    parser.add_argument('--no-val',
                        action='store_true',
                        default=False,
                        help='skip validation during training')

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
    parser.add_argument('--path_pretrained_model',
                        type=str,
                        default=None)
    parser.add_argument('--loss_type',
                        type=str,
                        default='dice')
    parser.add_argument('--lr_scheduler',
                        type=str,
                        default='step')

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

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 200,
            'pascal': 50,
            'isi': 50,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
            'isi': 0.01,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

    if args.checkname is None:
        args.checkname = 'deeplab-' + str(args.backbone)
    return args


def main():
    args = get_args()
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)

    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()


if __name__ == "__main__":
    main()

# import argparse
# import os
# import numpy as np
# from tqdm import tqdm
#
# import torch
# # import torchsummary
#
# from mypath import Path
# from datasets import make_data_loader
# from models.modeling.sync_batchnorm.replicate import patch_replication_callback
# from models.modeling.deeplab_multiview import DeepLabMultiView
# from utils.loss import SegmentationLosses
# from utils.calculate_weights import calculate_weigths_labels
# from utils.lr_scheduler import LR_Scheduler
# from utils.saver import Saver
# from utils.summaries import TensorboardSummary
# from utils.metrics import Evaluator
#
# import models.convcrf as convcrf
# from datasets.multiview_info import *
#
# import matplotlib.pyplot as plt
# from torchvision.utils import make_grid
# from datasets.utils import decode_seg_map_sequence
#
#
# def onehot(targets, num_classes):
#     """Origin: https://github.com/moskomule/mixup.pytorch
#     convert index tensor into onehot tensor
#     :param targets: index tensor
#     :param num_classes: number of classes
#     """
#     # assert isinstance(targets, torch.LongTensor)
#     return torch.zeros(targets.size()[0], num_classes).scatter_(1, targets.view(-1, 1), 1)
#
#
# def mixup(inputs, targets, num_classes, alpha=0.4):
#     """Mixup on 1x32x32 mel-spectrograms.
#     """
#     s = inputs.size()[0]
#     weight = torch.Tensor(np.random.beta(alpha, alpha, s))
#     index = np.random.permutation(s)
#     x1, x2 = inputs, inputs[index, :, :, :]
#     y1, y2 = onehot(targets, num_classes), onehot(targets[index,], num_classes)
#     weight = weight.view(s, 1, 1, 1)
#     inputs = weight * x1 + (1 - weight) * x2
#     weight = weight.view(s, 1)
#     targets = weight * y1 + (1 - weight) * y2
#     return inputs, targets
#
#
# class Trainer(object):
#     def __init__(self, args):
#         self.args = args
#
#         # Define Saver
#         self.saver = Saver(args)
#         self.saver.save_experiment_config()
#         # Define Tensorboard Summary
#         self.summary = TensorboardSummary(self.saver.experiment_dir)
#         self.writer = self.summary.create_summary()
#
#         # Define Dataloader
#         kwargs = {'num_workers': args.workers, 'pin_memory': True}
#         self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)
#
#         # Define network
#         model = DeepLabMultiView(num_classes=self.nclass,
#                                  backbone=args.backbone,
#                                  output_stride=args.out_stride,
#                                  sync_bn=args.sync_bn,
#                                  freeze_bn=args.freeze_bn)
#
#         self.crf = None
#         if args.GaussCrf:
#             if self.args.TrainCrf:
#                 conf = convcrf.isi_trainable_conf
#             else:
#                 conf = convcrf.isi_untrainable_conf
#
#             self.crf = convcrf.GaussCRF(conf=conf, shape=(513, 513), nclasses=self.nclass)
#
#         # load pretrain model
#         if args.path_pretrained_model is not None:
#             # model.load_state_dict(torch.load(args.path_pretrained_model)['state_dict'])
#
#             pretrained_dict = torch.load(args.path_pretrained_model)['state_dict']
#             model_dict = model.deeplab.state_dict()
#             # filter out uncessary keys
#             pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#             print(len(pretrained_dict))
#
#             # overwrite existing entries
#             model_dict.update(pretrained_dict)
#             # load the new state dict
#             model.deeplab.load_state_dict(model_dict)
#
#             # fix parameters
#             for param in model.deeplab.parameters():
#                 param.requires_grad = False
#         else:
#             # define different learning rate
#             train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
#                             {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]
#             # Define Optimizer
#             optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
#                                         weight_decay=args.weight_decay, nesterov=args.nesterov)
#
#         # print model parameter
#         # torchsummary.summary(model.cuda(), input_size=(4, 7, 287, 352), batch_size=1)
#
#         # Define Optimizer
#         optimizer = torch.optim.SGD(model.merger.parameters(), lr=args.lr, momentum=args.momentum,
#                                     weight_decay=args.weight_decay, nesterov=args.nesterov)
#
#         # # check if layers are fixed
#         # for name, param in model.named_parameters():
#         #     print(name, param.requires_grad)
#
#         # Define Criterion
#         # whether to use class balanced weights
#         if args.use_balanced_weights:
#             classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset + '_classes_weights.npy')
#             if os.path.isfile(classes_weights_path):
#                 weight = np.load(classes_weights_path)
#             else:
#                 weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
#             weight = torch.from_numpy(weight.astype(np.float32))
#         else:
#             weight = None
#         self.criterion = SegmentationLosses(weight=torch.ones([self.nclass, 1]).cuda(), cuda=args.cuda).build_loss(
#             mode=args.loss_type)
#         self.model, self.optimizer = model, optimizer
#
#         # Define Evaluator
#         self.evaluator = Evaluator(self.nclass)
#         self.evaluator_crf = Evaluator(self.nclass)
#         # Define lr scheduler
#         self.scheduler = LR_Scheduler(args.lr_scheduler,
#                                       args.lr,
#                                       args.epochs,
#                                       len(self.train_loader))
#
#         # Using cuda
#         if args.cuda:
#             self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
#             patch_replication_callback(self.model)
#             self.model = self.model.cuda()
#
#             if args.GaussCrf:
#                 self.crf = torch.nn.DataParallel(self.crf, device_ids=self.args.gpu_ids)
#                 patch_replication_callback(self.crf)
#                 self.crf = self.crf.cuda()
#
#         # Resuming checkpoint
#         self.best_pred = 0.0
#         if args.resume is not None:
#             print("loading checkpoint ...")
#             if not os.path.isfile(args.resume):
#                 raise RuntimeError(f"=> no checkpoint found at '{args.resume}'")
#             checkpoint = torch.load(args.resume)
#             args.start_epoch = checkpoint['epoch']
#             if args.cuda:
#                 self.model.module.load_state_dict(checkpoint['state_dict'])
#                 if args.GaussCrf and 'crf_state_dict' in checkpoint:
#                     if checkpoint['crf_state_dict'] is not None:
#                         print("loading crf")
#                         self.crf.module.load_state_dict(checkpoint['crf_state_dict'])
#             else:
#                 self.model.load_state_dict(checkpoint['state_dict'])
#                 if args.GaussCrf and 'crf_state_dict' in checkpoint:
#                     if checkpoint['crf_state_dict'] is not None:
#                         print("loading crf")
#                         self.crf.load_state_dict(checkpoint['crf_state_dict'])
#             if not args.ft:
#                 self.optimizer.load_state_dict(checkpoint['optimizer'])
#             self.best_pred = checkpoint['best_pred']
#             print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
#
#         # Clear start epoch if fine-tuning
#         if args.ft:
#             args.start_epoch = 0
#
#     def training(self, epoch):
#         train_loss = 0.0
#         self.model.train()
#         if self.args.TrainCrf:
#             self.crf.train()
#         tbar = tqdm(self.train_loader)
#         num_img_tr = len(self.train_loader)
#
#         for i, sample in enumerate(tbar):
#             image, target, pc = sample['image'], sample['label'], sample['pointcloud']
#             if image.shape[0] == 1:  # will fail otherwise because cannot have a batch size of 1
#                 continue
#
#             if self.args.cuda:
#                 image, target, pc = image.cuda(), target.cuda(), pc.cuda()
#
#             if self.args.loss_type == 'cem':
#                 image, target = mixup(image, onehot(target, 11), 11, alpha=0.4)
#
#             self.scheduler(self.optimizer, i, epoch, self.best_pred)
#             self.optimizer.zero_grad()
#
#             # go through network
#             output_singleview, output_multiview = self.model(image, pc, target)
#             if self.crf is not None and self.args.TrainCrf:
#                 output_multiview = self.crf(output_multiview, image)
#
#             # show image
#             # grid_image = make_grid(image[0, :, :, :, :][:4].clone().cpu().data, 4, normalize=True)
#             # plt.imshow(np.transpose(grid_image, (1, 2, 0)))
#             # plt.show()
#             # hard_label = torch.max(output[0, :, :, :, :][:4], 1)[1]
#             # print(hard_label.max())
#             # print(hard_label.min())
#             # grid_image = make_grid(
#             #     decode_seg_map_sequence(hard_label.clone().cpu().numpy(),
#             #                             dataset='isi'), 4, normalize=False, range=(0, 255))
#             # plt.imshow(np.transpose(grid_image, (1, 2, 0)))
#             # plt.show()
#             # grid_image = make_grid(
#             #     decode_seg_map_sequence(torch.squeeze(target[0, :, :, :][:4], 1).clone().cpu().numpy(),
#             #                             dataset='isi'), 4, normalize=False, range=(0, 255))
#             # plt.imshow(np.transpose(grid_image, (1, 2, 0)))
#             # plt.show()
#
#             # calculate loss
#             loss = 0.0
#             for camid in range(CAM_NUM):
#                 loss = loss + self.criterion(output_multiview[:, camid, :, :, :], target[:, camid, :, :])
#             loss.backward()
#
#             # # print the gradient
#             # for name, param in self.model.module.merger.named_parameters():
#             #     print(name, param.grad)
#
#             # backprop
#             self.optimizer.step()
#             train_loss += loss.item()
#             tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
#             self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)
#
#             # Show 10 * 3 inference results each epoch
#             if i % (num_img_tr // 2) == 0:
#                 global_step = i + num_img_tr * epoch
#                 self.summary.visualize_image(self.writer, self.args.dataset, image[0, :, :, :, :],
#                                              target[0, :, :, :], output_singleview[0, :, :, :, :],
#                                              output_multiview[0, :, :, :, :],
#                                              global_step)
#
#         self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
#         print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
#         print('Loss: %.3f' % train_loss)
#
#         if self.args.no_val:
#             # save checkpoint every epoch
#             is_best = False
#             self.saver.save_checkpoint({
#                 'epoch': epoch + 1,
#                 'state_dict': self.model.module.state_dict(),
#                 'optimizer': self.optimizer.state_dict(),
#                 'best_pred': self.best_pred,
#                 'crf_state_dict': self.crf.module.state_dict() if self.crf is not None else None,
#             }, is_best)
#
#     def validation(self, epoch):
#         self.model.eval()
#         # self.crf.eval()
#         self.evaluator.reset()
#         if self.crf is not None:
#             self.crf.eval()
#             self.evaluator_crf.reset()
#         tbar = tqdm(self.val_loader, desc='\r')
#         test_loss = 0.0
#         for i, sample in enumerate(tbar):
#             image, target, pc = sample['image'], sample['label'], sample['pointcloud']
#             if self.args.cuda:
#                 image, target, pc = image.cuda(), target.cuda(), pc.cuda()
#
#             with torch.no_grad():
#                 output_singleview, output_multiview = self.model(image, pc, target)
#                 if self.crf is not None:
#                     # self.crf.module.CRF.npixels = (1080, 1080)
#                     # self.crf.module.CRF.height = 1080
#                     # self.crf.module.CRF.width = 1080
#                     # print(f"output, {output.shape}, img:{image.shape}")
#                     output_crf = self.crf.forward(unary=output_multiview, img=image)
#
#                 # calculate loss
#                 loss = 0.0
#                 for camid in range(CAM_NUM):
#                     loss = loss + self.criterion(output_multiview[:, camid, :, :, :], target[:, camid, :, :])
#
#             self.item = loss.item()
#             test_loss += self.item
#             tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
#
#             for camid in range(CAM_NUM):
#                 pred_tmp = output_multiview[:, camid, :, :, :].data.cpu().numpy()
#                 target_tmp = target[:, camid, :, :].cpu().numpy()
#                 pred_tmp = np.argmax(pred_tmp, axis=1)
#                 # Add batch sample into evaluator
#                 self.evaluator.add_batch(target_tmp, pred_tmp)
#
#             if self.crf is not None:
#                 pred_crf = np.argmax(output_crf.data.cpu().numpy(), axis=1)
#                 self.evaluator_crf.add_batch(target, pred_crf)
#
#         # Fast test during the training
#         Acc = self.evaluator.Pixel_Accuracy()
#         Acc_class = self.evaluator.Pixel_Accuracy_Class()
#         mIoU = self.evaluator.Mean_Intersection_over_Union()
#         FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
#         IoU = self.evaluator.Intersection_over_Union()
#         self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
#         self.writer.add_scalar('val/mIoU', mIoU, epoch)
#         self.writer.add_scalar('val/Acc', Acc, epoch)
#         self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
#         self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
#         print('Validation:')
#         print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
#         print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
#         for i_class, class_IoU in enumerate(IoU):
#             print("{} class, IoU is {:1.2f}\n".format(i_class, class_IoU))
#         print('Loss: %.3f' % test_loss)
#
#         if self.crf is not None:
#             # Fast test during the training
#             Acc_crf = self.evaluator_crf.Pixel_Accuracy()
#             Acc_class_crf = self.evaluator_crf.Pixel_Accuracy_Class()
#             mIoU_crf = self.evaluator_crf.Mean_Intersection_over_Union()
#             FWIoU_crf = self.evaluator_crf.Frequency_Weighted_Intersection_over_Union()
#             self.writer.add_scalar('val/mIoU_crf', mIoU_crf, epoch)
#             self.writer.add_scalar('val/Acc_crf', Acc_crf, epoch)
#             self.writer.add_scalar('val/Acc_class_crf', Acc_class_crf, epoch)
#             self.writer.add_scalar('val/fwIoU_crf', FWIoU_crf, epoch)
#             print('Validation_crf:')
#             print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc_crf, Acc_class_crf, mIoU_crf, FWIoU_crf))
#
#         new_pred = mIoU
#         if new_pred > self.best_pred:
#             is_best = True
#             self.best_pred = new_pred
#             self.saver.save_checkpoint({
#                 'epoch': epoch + 1,
#                 'state_dict': self.model.module.state_dict(),
#                 'optimizer': self.optimizer.state_dict(),
#                 'best_pred': self.best_pred,
#                 'crf_state_dict': self.crf.module.state_dict() if self.crf is not None else None,
#             }, is_best)
#
#
# #  pred = gausscrf.forward(unary=unary_var, img=img_var)
# def get_args():
#     parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
#     parser.add_argument('--backbone',
#                         type=str,
#                         default='resnet',
#                         choices=['resnet', 'xception', 'drn', 'mobilenet'],
#                         help='backbone name (default: resnet)')
#     parser.add_argument('--out-stride',
#                         type=int,
#                         default=16,
#                         help='network output stride (default: 8)')
#     parser.add_argument('--dataset',
#                         type=str,
#                         default='isi',
#                         choices=['pascal', 'coco', 'cityscapes', 'isi_multiview'],
#                         help='dataset name (default: isi)')
#     parser.add_argument('--use-sbd',
#                         action='store_true',
#                         default=True,
#                         help='whether to use SBD dataset (default: True)')
#     parser.add_argument('--workers',
#                         type=int,
#                         default=4,
#                         metavar='N',
#                         help='dataloader threads')
#     parser.add_argument('--base-size',
#                         type=int,
#                         default=513,
#                         help='base image size')
#     parser.add_argument('--crop-size',
#                         type=int,
#                         default=513,
#                         help='crop image size')
#     parser.add_argument('--sync-bn',
#                         type=bool,
#                         default=None,
#                         help='whether to use sync bn (default: auto)')
#     parser.add_argument('--freeze-bn',
#                         type=bool,
#                         default=False,
#                         help='whether to freeze bn parameters (default: False)')
#     parser.add_argument('--loss-type',
#                         type=str,
#                         default='ce',
#                         choices=['ce', 'focal', 'cem'],
#                         help='loss func type (default: ce)')
#     # training hyper params
#     parser.add_argument('--epochs',
#                         type=int,
#                         default=None,
#                         metavar='N',
#                         help='number of epochs to train (default: auto)')
#     parser.add_argument('--start_epoch',
#                         type=int, default=0,
#                         metavar='N',
#                         help='start epochs (default:0)')
#     parser.add_argument('--batch-size',
#                         type=int,
#                         default=None,
#                         metavar='N',
#                         help='input batch size for training (default: auto)')
#     parser.add_argument('--test-batch-size',
#                         type=int,
#                         default=None,
#                         metavar='N',
#                         help='input batch size for testing (default: auto)')
#     parser.add_argument('--use-balanced-weights',
#                         action='store_true',
#                         default=False,
#                         help='whether to use balanced weights (default: False)')
#     # optimizer params
#     parser.add_argument('--lr',
#                         type=float,
#                         default=None,
#                         metavar='LR',
#                         help='learning rate (default: auto)')
#     parser.add_argument('--lr-scheduler',
#                         type=str,
#                         default='poly',
#                         choices=['poly', 'step', 'cos'],
#                         help='lr scheduler mode: (default: poly)')
#     parser.add_argument('--momentum',
#                         type=float,
#                         default=0.9,
#                         metavar='M',
#                         help='momentum (default: 0.9)')
#     parser.add_argument('--weight-decay',
#                         type=float,
#                         default=5e-4,
#                         metavar='M',
#                         help='w-decay (default: 5e-4)')
#     parser.add_argument('--nesterov',
#                         action='store_true',
#                         default=False,
#                         help='whether use nesterov (default: False)')
#     # cuda, seed and logging
#     parser.add_argument('--no-cuda',
#                         action='store_true',
#                         default=False,
#                         help='disables CUDA training')
#     parser.add_argument('--gpu-ids',
#                         type=str,
#                         default='0',
#                         help='use which gpu to train, must be a comma-separated list of integers only (default=0)')
#     parser.add_argument('--seed',
#                         type=int,
#                         default=1,
#                         metavar='S',
#                         help='random seed (default: 1)')
#     # checking point
#     parser.add_argument('--resume',
#                         type=str,
#                         default=None,
#                         help='put the path to resuming file if needed')
#     parser.add_argument('--checkname',
#                         type=str,
#                         default=None,
#                         help='set the checkpoint name')
#     # finetuning pre-trained models
#     parser.add_argument('--ft',
#                         action='store_true',
#                         default=False,
#                         help='finetuning on a different dataset')
#     # evaluation option
#     parser.add_argument('--eval-interval',
#                         type=int,
#                         default=1,
#                         help='evaluation interval (default: 1)')
#     parser.add_argument('--no-val',
#                         action='store_true',
#                         default=False,
#                         help='skip validation during training')
#
#     # ISI
#     parser.add_argument('--dataset_dir',
#                         type=str,
#                         default='/home/deepsight2/development/data/rgb',
#                         help='put the path to dataset root dir')
#     parser.add_argument('--GaussCrf',
#                         action='store_true',
#                         default=False,
#                         help='Add GaussCRF at the end of the model')
#     parser.add_argument('--TrainCrf',
#                         action='store_true',
#                         default=False,
#                         help='Train GaussCRF at the end of the model')
#     parser.add_argument('--crf_start_epoch',
#                         type=int,
#                         default=0,
#                         help='Epoch at which to start training the CRF (not stable if trained from the begining!)')
#     parser.add_argument('--path_pretrained_model',
#                         type=str,
#                         default=None)
#     parser.add_argument('--loss_type',
#                         type=str,
#                         default='dice')
#     parser.add_argument('--lr_scheduler',
#                         type=str,
#                         default='step')
#
#     args = parser.parse_args()
#
#     args.cuda = not args.no_cuda and torch.cuda.is_available()
#     if args.cuda:
#         try:
#             args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
#         except ValueError:
#             raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')
#
#     if args.sync_bn is None:
#         if args.cuda and len(args.gpu_ids) > 1:
#             args.sync_bn = True
#         else:
#             args.sync_bn = False
#
#     # default settings for epochs, batch_size and lr
#     if args.epochs is None:
#         epoches = {
#             'coco': 30,
#             'cityscapes': 200,
#             'pascal': 50,
#             'isi': 50,
#         }
#         args.epochs = epoches[args.dataset.lower()]
#
#     if args.batch_size is None:
#         args.batch_size = 4 * len(args.gpu_ids)
#
#     if args.test_batch_size is None:
#         args.test_batch_size = args.batch_size
#
#     if args.lr is None:
#         lrs = {
#             'coco': 0.1,
#             'cityscapes': 0.01,
#             'pascal': 0.007,
#             'isi': 0.01,
#         }
#         args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size
#
#     if args.checkname is None:
#         args.checkname = 'deeplab-' + str(args.backbone)
#     return args
#
#
# def main():
#     args = get_args()
#     print(args)
#     torch.manual_seed(args.seed)
#     trainer = Trainer(args)
#     print('Starting Epoch:', trainer.args.start_epoch)
#     print('Total Epoches:', trainer.args.epochs)
#
#     for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
#         trainer.training(epoch)
#         if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
#             trainer.validation(epoch)
#
#     trainer.writer.close()
#
#
# if __name__ == "__main__":
#     main()
