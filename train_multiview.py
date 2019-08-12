import argparse
import os
import numpy as np
from tqdm import tqdm
import random

import torch
import torch.autograd as autograd

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

from models.network_initialization import init_net

from models.modeling.discriminator import Discriminator, onehot

CAM_NUM = 4
HEIGHT = 287
WIDTH = 352


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
                                 freeze_bn=args.freeze_bn,
                                 unet_size=args.unet_size,
                                 separable_conv=args.separable_conv)
        model.merger = init_net(model.merger, type="kaiming", activation_mode='relu', distribution='normal')

        if args.adversarial_loss:
            print('define discriminator')
            discriminator = Discriminator(input_nc=self.nclass,
                                          img_height=287,
                                          img_width=352,
                                          filter_base=16,
                                          num_block=4,
                                          n_iter=args.n_critic,
                                          generator_loss_weight=self.args.generator_loss_weight,
                                          lr_ratio=1,
                                          gp_weigth=1)
            discriminator = init_net(discriminator, type='kaiming', activation_mode='relu', distribution='normal')

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
        if args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.merger.parameters(), lr=args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay, nesterov=args.nesterov)
        elif args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.merger.parameters(), lr=args.lr,
                                         weight_decay=args.weight_decay)
        else:
            optimizer = None

        if args.adversarial_loss:
            print('add optimizer for discriminator')
            optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr * discriminator.lr_ratio,
                                           weight_decay=args.weight_decay)

            self.discriminator, self.optimizer_D = discriminator, optimizer_D

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
        self.define_evaluators()
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

            if args.adversarial_loss:
                self.discriminator = torch.nn.DataParallel(self.discriminator, device_ids=self.args.gpu_ids)
                patch_replication_callback(self.discriminator)
                self.discriminator = self.discriminator.cuda()

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

                if self.args.adversarial_loss:
                    self.discriminator.module.load_state_dict(checkpoint['discriminator_state_dict'])

            else:
                self.model.load_state_dict(checkpoint['state_dict'])

                if self.args.adversarial_loss:
                    self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def define_evaluators(self):
        if self.args.dataset == "isi_intensity":
            weights = np.array([1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1]) > 0.5
        elif self.args.dataset == 'isi_multiview':
            weights = self.args.skip_weights > 0.5
        elif self.args.dataset == 'isi_multiview_2018':
            weights = self.args.skip_weights > 0.5
        elif (
                self.args.dataset == "isi_rgb" or self.args.dataset == "isi_rgb_temporal") and self.args.skip_classes is not None:
            weights = self.args.skip_weights > 0.5
        else:
            weights = None
        print('Weights: ', weights)
        self.evaluator = Evaluator(self.nclass, weights)

    def training(self, epoch):
        train_loss = 0.0
        train_d_loss = 0.0
        train_g_loss = 0.0
        self.model.train()

        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)

        if self.args.adversarial_loss:
            train_d = False
            mse_loss = torch.nn.MSELoss()

        for i, sample in enumerate(tbar):
            image, target, pc = sample['image'], sample['label'], sample['pointcloud']

            if self.args.cuda:
                image, target, pc = image.cuda(), target.cuda(), pc.cuda()

            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            if self.args.adversarial_loss:
                self.optimizer_D.zero_grad()

            # go through network
            output_single_view, output_merger, target = self.model(image, pc, target)

            # pixel loss
            loss = 0.0
            for camid in range(CAM_NUM):
                # pixel loss
                pixel_loss = self.criterion(output_merger[:, camid, :, :, :], target[:, camid, :, :])
                loss += pixel_loss

            # adversarial loss
            if self.args.adversarial_loss:
                # switch training discriminator and network
                train_d = not train_d if (i + num_img_tr * epoch) % self.discriminator.module.n_iter == 0 else train_d

                loss_D = 0.0
                loss_G = 0.0

                # train discriminator
                for camid in range(CAM_NUM):
                    # fake
                    fake_input = torch.softmax(output_merger[:, camid, :, :, :], dim=1)
                    fake_validity = self.discriminator(fake_input)
                    # real
                    real_input = target[:, camid, :, :]
                    real_input = onehot(real_input, self.nclass)
                    # real_input = label_smoothing(real_input, epoch, self.args.epochs)
                    real_validity = self.discriminator(real_input)
                    # mean
                    mean_validity_real = torch.mean(real_validity, dim=0, keepdim=True).expand_as(
                        fake_validity).detach()
                    mean_validity_fake = torch.mean(fake_validity, dim=0, keepdim=True).expand_as(
                        real_validity).detach()

                    gradient_penality = False
                    if gradient_penality:
                        gp = compute_gradient_penality(self.discriminator, real_input, fake_input)

                    # discriminator
                    if train_d:
                        # discriminator loss
                        real_loss = mse_loss(real_validity - mean_validity_fake, torch.ones_like(real_validity))
                        fake_loss = mse_loss(fake_validity - mean_validity_real,
                                             torch.ones_like(fake_validity) * (- 1.0))
                        loss_D += real_loss + fake_loss  # + gp
                        # print('gp ', gp)

                    # train network
                    else:
                        # discriminator loss
                        real_loss = mse_loss(real_validity - mean_validity_fake,
                                             torch.ones_like(real_validity) * (- 1.0))
                        fake_loss = mse_loss(fake_validity - mean_validity_real, torch.ones_like(fake_validity))
                        loss_G += real_loss + fake_loss

            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            if self.args.adversarial_loss:
                if train_d:
                    # backprop
                    loss_D.backward()
                    self.optimizer_D.step()
                    self.writer.add_scalar('train/discriminator_loss', loss_D.item(), i + num_img_tr * epoch)
                else:
                    loss += self.discriminator.module.generator_loss_weight * loss_G
                    # backprop
                    loss.backward()
                    self.optimizer.step()
                    self.writer.add_scalar('train/generator_loss',
                                           self.discriminator.module.generator_loss_weight * loss_G.item(),
                                           i + num_img_tr * epoch)

            # Show 10 * 3 inference results each epoch
            if i % (num_img_tr // 2) == 0:
                global_step = i + num_img_tr * epoch
                self.summary.visualize_image_mulitview(self.writer, self.args.dataset, image[0, :, :, :, :],
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
                'discriminator_state_dict': self.discriminator.module.state_dict() if self.args.adversarial_loss else None,
            }, is_best)

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()

        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target, pc = sample['image'], sample['label'], sample['pointcloud']
            if self.args.cuda:
                image, target, pc = image.cuda(), target.cuda(), pc.cuda()

            with torch.no_grad():
                output_single_view, output_merger, target = self.model(image, pc, target)

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

        self.summary.visualize_validation_mulitview(self.writer, self.args.dataset, image[0, :, :, :, :],
                                                    target[0, :, :, :], output_single_view[0, :, :, :, :],
                                                    output_merger[0, :, :, :, :],
                                                    epoch)

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
                'discriminator_state_dict': self.discriminator.module.state_dict() if self.args.adversarial_loss else None,
            }, is_best)


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
                        choices=['pascal', 'coco', 'cityscapes', 'isi_intensity', 'isi_multiview',
                                 'isi_multiview_2018'],
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
                        choices=['ce', 'focal', 'cem', 'dice'],
                        help='loss func type (default: ce)')
    parser.add_argument('--adversarial_loss',
                        type=bool,
                        default=False,
                        help='if use adversarial loss for shape regulation or not')
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
    parser.add_argument('--optimizer',
                        type=str,
                        default='Adam',
                        help='optimizer type: (default: Adam')
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
    parser.add_argument('--generator_loss_weight',
                        type=float,
                        default=0.0005)
    parser.add_argument('--unet_size',
                        type=str,
                        default='Medium')
    parser.add_argument('--skip_classes',
                        type=str,
                        default=None,
                        help='Classes to skip in the computation of the iou and the loss')
    parser.add_argument('--separable_conv',
                        action='store_true',
                        default=False,
                        help='separable convolution')
    parser.add_argument('--n_critic',
                        type=int,
                        default=2,
                        help='how many iterations to switch between generator and discriminator')

    args = parser.parse_args()

    # skip classes
    if args.skip_classes is not None:
        if args.dataset == 'isi_rgb' or args.dataset == 'isi_rgb_temporal':
            CLASSES = ['ortable',
                       'psc',
                       'vsc',
                       'human',
                       'cielinglight',
                       'mayostand',
                       'table',
                       'anesthesiacart',
                       'cannula',
                       'instrument']
            print(CLASSES)
            label_name_to_value = {x: i + 1 for i, x in enumerate(CLASSES)}
            weights = np.ones(len(CLASSES) + 1)
            for c in args.skip_classes.split(','):
                weights[label_name_to_value[c]] = 0
            args.skip_weights = weights

        elif args.dataset == 'isi_multiview':
            CLASSES = ['ortable', 'psc', 'vsc', 'human', 'cielinglight', 'floor', 'mayostand', 'table', 'chair', 'wall',
                       'anesthesiacart', 'cannula']
            label_name_to_value = {x: i + 1 for i, x in enumerate(CLASSES)}
            weights = np.ones(len(CLASSES) + 1)
            for c in args.skip_classes.split(','):
                weights[label_name_to_value[c]] = 0
            args.skip_weights = weights

        elif args.dataset == 'isi_multiview_2018':
            CLASSES = ['ortable', 'robot', 'human', 'cielinglight', 'floor', 'standtable', 'chair', 'wall',
                       'visioncart']
            label_name_to_value = {x: i + 1 for i, x in enumerate(CLASSES)}
            weights = np.ones(len(CLASSES) + 1)
            for c in args.skip_classes.split(','):
                weights[label_name_to_value[c]] = 0
            args.skip_weights = weights

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
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
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
