import argparse
import os
import random

import numpy as np
import torch
from tqdm import tqdm

import models.convcrf as convcrf
from datasets import make_data_loader
from models.modeling.deeplab import DeepLab
from models.modeling.discriminator import Discriminator, onehot
from models.modeling.sync_batchnorm.replicate import patch_replication_callback
from models.network_initialization import init_net
from utils.calculate_weights import calculate_weigths_labels
from utils.loss import SegmentationLosses
from utils.lr_scheduler import LR_Scheduler
from utils.metrics import Evaluator
from utils.saver import Saver
from utils.summaries import TensorboardSummary


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.prepare_saver()
        self.prepare_tensorboard()
        self.prepare_dataloader()

        self.model = DeepLab(num_classes=self.nclass,
                             backbone=args.backbone,
                             output_stride=args.out_stride,
                             sync_bn=args.sync_bn,
                             freeze_bn=args.freeze_bn)
        self.build_crf()
        self.define_optimizer()

        if args.adversarial_loss:
            self.build_adverserial_model()
        else:
            self.discriminator = None
            self.optimizer_D = None

        self.define_pixel_criterion()
        self.define_evaluators()

        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler,
                                      args.lr,
                                      args.epochs,
                                      len(self.train_loader))

        # Using cuda
        if self.args.cuda:
            self.model_on_cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            self.resume_model()

    def prepare_saver(self):
        self.saver = Saver(self.args)
        self.saver.save_experiment_config()

    def prepare_tensorboard(self):
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

    def prepare_dataloader(self):
        kwargs = {'num_workers': self.args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(self.args, **kwargs)

    def define_evaluators(self):
        if self.args.dataset == "isi_intensity":
            weights = np.array([1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1]) > 0.5

        if (self.args.dataset == "isi_rgb" or self.args.dataset =="isi_rgb_temporal") and self.args.skip_classes is not None:
            weights = self.args.skip_weights > 0.5
        else:
            weights = None

        self.evaluator = Evaluator(self.nclass, weights)
        self.evaluator_crf = Evaluator(self.nclass, weights)

    def define_optimizer(self):
        if self.args.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=self.args.lr,
                                             momentum=self.args.momentum,
                                             weight_decay=self.args.weight_decay,
                                             nesterov=self.args.nesterov)
        elif self.args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=self.args.lr,
                                              weight_decay=self.args.weight_decay)
        else:
            raise RuntimeError(f"Chosen optimizer is not usable !!!")

    def define_pixel_criterion(self):
        if self.args.use_balanced_weights:
            classes_weights_path = os.path.join(self.args.dataset_dir, 'classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(self.args.dataset_dir, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
            print(f"Classes weights : {weight}")

        elif self.args.skip_classes is not None:
            weight = torch.from_numpy(self.args.skip_weights.astype(np.float32))
            print(f"Classes weights : {weight}")
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=self.args.cuda).build_loss(mode=self.args.loss_type)

    def build_adverserial_model(self):
        self.adv_loss = torch.nn.MSELoss()
        self.generator_loss_weight = self.args.generator_loss_weight
        self.gradient_penalty_weight = self.args.gradient_penalty_weight
        self.n_critic = self.args.n_critic
        self.train_d = False
        print('Define discriminator')
        self.discriminator = Discriminator(input_nc=self.nclass,
                                           img_height=self.args.img_shape[0],
                                           img_width=self.args.img_shape[1],
                                           filter_base=16,
                                           n_iter=self.args.n_critic,
                                           generator_loss_weight=self.args.generator_loss_weight,
                                           lr_ratio=self.args.lr_ratio,
                                           gp_weigth=self.args.gradient_penalty_weight,
                                           num_block=self.args.discriminator_blocks)
        self.discriminator = init_net(self.discriminator,
                                      type='kaiming',
                                      activation_mode='relu',
                                      distribution='normal')
        print('Add optimizer for discriminator')
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(),
                                            lr=self.args.lr * self.args.lr_ratio,
                                            weight_decay=self.args.weight_decay)

    def build_crf(self):
        self.crf = None
        if self.args.GaussCrf:
            if self.args.TrainCrf:
                conf = convcrf.isi_trainable_conf
            else:
                conf = convcrf.isi_untrainable_conf

            if self.args.dataset == "isi_rgb":
                shape = (513, 513)
            elif self.args.dataset == "isi_intensity":
                shape = (287, 352)

            self.crf = convcrf.GaussCRF(conf=conf, shape=shape, nclasses=self.nclass)

    def model_on_cuda(self):
        self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
        patch_replication_callback(self.model)
        self.model = self.model.cuda()

        if self.args.GaussCrf:
            self.crf = torch.nn.DataParallel(self.crf, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.crf)
            self.crf = self.crf.cuda()

        if self.args.adversarial_loss:
            self.discriminator = torch.nn.DataParallel(self.discriminator, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.discriminator)
            self.discriminator = self.discriminator.cuda()

    def resume_model(self):
        if not os.path.isfile(self.args.resume):
            raise RuntimeError(f"=> no checkpoint found at '{self.args.resume}'")
        checkpoint = torch.load(self.args.resume)
        self.args.start_epoch = checkpoint['epoch']

        # Model + Discriminator
        if self.args.cuda:
            self.model.module.load_state_dict(checkpoint['state_dict'])
            if self.args.GaussCrf and 'crf_state_dict' in checkpoint:
                if checkpoint['crf_state_dict'] is not None:
                    self.crf.module.load_state_dict(checkpoint['crf_state_dict'])
            if self.args.adversarial_loss and 'discriminator_state_dict' in checkpoint:
                if checkpoint['discriminator_state_dict'] is not None:
                    self.discriminator.module.load_state_dict(checkpoint['discriminator_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['state_dict'])
            if self.args.GaussCrf and 'crf_state_dict' in checkpoint:
                if checkpoint['crf_state_dict'] is not None:
                    self.crf.load_state_dict(checkpoint['crf_state_dict'])
            if self.args.adversarial_loss and 'discriminator_state_dict' in checkpoint:
                if checkpoint['discriminator_state_dict'] is not None:
                    self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

        # Optimizers
        if self.args.ft:
            self.args.start_epoch = 0
            self.best_pred = 0.0
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.args.adversarial_loss and 'discriminator_optimizer_state_dict' in checkpoint:
                if checkpoint['discriminator_optimizer_state_dict'] is not None:
                    self.optimizer_D.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
            self.best_pred = checkpoint['best_pred']

        if self.args.print_ft:
            print(f"=> loaded checkpoint '{self.args.resume}' (epoch {checkpoint['epoch']})")

    def train_adverserial(self, output, target, i, num_img_tr, epoch, loss):
        self.optimizer_D.zero_grad()

        # switch training discriminator and network
        self.train_d = not self.train_d if (i + num_img_tr * epoch) % self.discriminator.module.n_iter == 0 else self.train_d

        loss_D = 0.0
        loss_G = 0.0

        # fake
        fake_input = torch.softmax(output, dim=1)
        fake_validity = self.discriminator(fake_input)
        # real
        real_input = target
        real_input = onehot(real_input, self.nclass)
        real_validity = self.discriminator(real_input)
        # mean
        mean_validity_real = torch.mean(real_validity, dim=0, keepdim=True).expand_as(fake_validity).detach()
        mean_validity_fake = torch.mean(fake_validity, dim=0, keepdim=True).expand_as(real_validity).detach()

        # gradient_penality = False
        # if gradient_penality:
        #     gp = compute_gradient_penality(self.discriminator, real_input, fake_input)

        # discriminator
        if self.train_d:
            # discriminator loss
            real_loss = self.adv_loss(real_validity - mean_validity_fake, torch.ones_like(real_validity))
            fake_loss = self.adv_loss(fake_validity - mean_validity_real, torch.ones_like(fake_validity) * (- 1.0))
            loss_D += real_loss + fake_loss  # + gp

            # backprop
            loss_D.backward()
            self.optimizer_D.step()
            self.writer.add_scalar('train/discriminator_loss', loss_D.item(), i + num_img_tr * epoch)
        # train network
        else:
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # discriminator loss
            real_loss = self.adv_loss(real_validity - mean_validity_fake, torch.ones_like(real_validity) * (- 1.0))
            fake_loss = self.adv_loss(fake_validity - mean_validity_real, torch.ones_like(fake_validity))
            loss_G += real_loss + fake_loss

            adv_loss = self.discriminator.module.generator_loss_weight * loss_G

            loss += adv_loss

            # backprop
            loss.backward()
            self.optimizer.step()
            self.writer.add_scalar('train/generator_loss', adv_loss.item(), i + num_img_tr * epoch)

    def save(self, epoch, is_best):
        self.checkpoint = self.saver.save_checkpoint(
            {'epoch': epoch + 1, 'state_dict': self.model.module.state_dict(), 'optimizer': self.optimizer.state_dict(),
             'best_pred': self.best_pred,
             'crf_state_dict': self.crf.module.state_dict() if self.crf is not None else None,
             'discriminator_state_dict': self.discriminator.module.state_dict() if self.discriminator is not None else None,
             'discriminator_optimizer_state_dict': self.optimizer_D.state_dict() if self.optimizer_D is not None else None},
            is_best)

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        if self.args.TrainCrf:
            self.crf.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if image.shape[0] == 1:  # will fail otherwise because cannot have a batch size of 1
                continue
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            if self.crf is not None and self.args.TrainCrf:
                output = self.crf(output, image)
            loss = self.criterion(output, target)
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            if self.args.adversarial_loss:
                self.train_adverserial(output, target, i, num_img_tr, epoch, loss)
            else:
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
                self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # Show 10 * 3 inference results each epoch
            if i % (num_img_tr // 2) == 0:
                global_step = i + num_img_tr * epoch
                self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        if self.args.no_val:
            # save checkpoint every epoch
            self.save(epoch, is_best=False)

    def measure_performance(self, evaluator, epoch, total_loss, image, i, name=''):
        Acc = evaluator.Pixel_Accuracy()
        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar(f'val/total_loss_epoch{name}', total_loss, epoch)
        self.writer.add_scalar(f'val/mIoU{name}', mIoU, epoch)
        self.writer.add_scalar(f'val/Acc{name}', Acc, epoch)
        self.writer.add_scalar(f'val/Acc_class{name}', Acc_class, epoch)
        self.writer.add_scalar(f'val/fwIoU{name}', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % total_loss)

        return mIoU

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()

        if self.crf is not None:
            self.crf.eval()
            self.evaluator_crf.reset()

        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
                if self.crf is not None:
                    output_crf = self.crf.forward(unary=output, img=image)
            loss = self.criterion(output_crf if self.crf is not None else output, target)
            test_loss += loss.item()

            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))

            if i == 0:
                self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, epoch, name="_val")

            target = target.cpu().numpy()
            pred = np.argmax(output.data.cpu().numpy(), axis=1)

            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)
            if self.crf is not None:
                pred_crf = np.argmax(output_crf.data.cpu().numpy(), axis=1)
                self.evaluator_crf.add_batch(target, pred_crf)

        # Fast test during the training
        new_pred = self.measure_performance(self.evaluator, epoch, test_loss, image, i)

        if self.crf is not None:
            new_pred = self.measure_performance(self.evaluator_crf, epoch, test_loss, image, i, name="_crf")

        if new_pred > self.best_pred:
            self.best_pred = new_pred
            self.save(epoch, is_best=True)


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
                        default='isi',
                        choices=['pascal', 'coco', 'cityscapes', 'isi_rgb', 'isi_intensity', 'isi_depth', 'isi_rgb_temporal'],
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
    parser.add_argument('--cuda_visible_devices',
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
    parser.add_argument('--print_ft',
                        type=bool,
                        default=True,
                        help='print finetuning on a different dataset')
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
    parser.add_argument('--skip_classes',
                        type=str,
                        default=None,
                        help='Classes to skip in the computation of the iou and the loss')
    parser.add_argument('--img_shape',
                        type=str,
                        default="513,513",
                        help='Image shape')
    parser.add_argument('--hd',
                        action='store_true',
                        default=False,
                        help='Add GaussCRF at the end of the model')

    # Adversarial loss
    parser.add_argument('--adversarial_loss',
                        action='store_true',
                        default=False,
                        help='if use adversarial loss for shape regulation or not')
    parser.add_argument('--gradient_penalty_weight',
                        type=float,
                        default=1.0,
                        help='learning rate (default: 1.0)')
    parser.add_argument('--generator_loss_weight',
                        type=float,
                        default=0.0005,
                        help='generator_loss_weight (default: 0.0005)')
    parser.add_argument('--n_critic',
                        type=int,
                        default=2,
                        help='n_critic (default: 2)')
    parser.add_argument('--lr_ratio',
                        type=float,
                        default=1/7,
                        help='lr_ratio (default: 1/7)')
    parser.add_argument('--discriminator_blocks',
                        type=int,
                        default=4,
                        help='discriminator_blocks (default: 4)')

    # Temporal
    parser.add_argument('--separate_spatial_model_path',
                        type=str,
                        default=None,
                        help='Path to the spatial model when pretrained seperatly')
    parser.add_argument('--svc_kernel_size',
                        type=int,
                        default=11,
                        help='svc_kernel_size (default: 11)')
    parser.add_argument('--train_distance',
                        type=int,
                        default=1000,
                        help='train_distance (default: 1000)')
    parser.add_argument('--flow',
                        action='store_true',
                        default=False,
                        help='Use feature flow for the kernel weights predictor (default: False)')

    # Demo
    parser.add_argument('--demo_camera',
                        action='store_true',
                        default=False,
                        help='Use camera for live demo (default: False)')
    parser.add_argument('--demo_img_folder',
                        type=str,
                        default=None,
                        help='List of folders containing image on which to do inference')
    parser.add_argument('--demo_video_path',
                        type=str,
                        default=None,
                        help='path to a video to be processed')
    parser.add_argument('--demo_video_output',
                        type=str,
                        default=None,
                        help='path to a video to be processed')
    parser.add_argument('--demo_temporal',
                        action='store_true',
                        default=False,
                        help='Use temporal model (default: False)')
    parser.add_argument('--demo_frame_fixed_schedule',
                        type=int,
                        default=10,
                        help='train_distance (default: 10)')

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
            'isi_rgb': 100,
            'isi_intensity': 100,
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
            'isi_rgb': 0.01,
            'isi_intensity': 0.01
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

    if args.checkname is None:
        args.checkname = 'deeplab-' + str(args.backbone)

    if args.skip_classes is not None:
        print(args.dataset, args.dataset == 'isi_rgb')
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

    args.img_shape = [int(i) for i in args.img_shape.split(',')]

    return args


def seed(seed=1234):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = get_args()
    seed(args.seed)
    print(args)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    main()
