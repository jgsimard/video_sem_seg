import os
import os

import numpy as np
import torch
from tqdm import tqdm

from models.afp import LowLatencyModel
# from models.modeling.deeplab import *
from models.modeling.deeplab import DeepLab
from train import Trainer, get_args
from utils.lr_scheduler import LR_Scheduler

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



class TemporalTrainer(Trainer):
    def __init__(self, args):
        self.args = args
        self.prepare_saver()
        self.prepare_tensorboard()
        self.prepare_dataloader()

        self.spatial_model = DeepLab(num_classes=self.nclass,
                                     backbone=args.backbone,
                                     output_stride=args.out_stride,
                                     sync_bn=args.sync_bn,
                                     freeze_bn=True)
        # Fix deeplab as the features extractor
        for param in self.spatial_model.parameters():
            param.requires_grad = False

        self.temporal_model = LowLatencyModel(self.spatial_model, kernel_size=self.args.svc_kernel_size, flow=self.args.flow)
        self.define_optimizer()

        if args.adversarial_loss:
            self.build_adverserial_model()
        else:
            self.discriminator = None
            self.optimizer_D = None

        self.define_pixel_criterion()
        self.define_evaluators()
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(self.train_loader))

        if self.args.cuda:
            self.model_on_cuda()

        if self.args.separate_spatial_model_path is not None:
            self.load_spatial_model_separately()

        self.best_pred = 0.0
        if args.resume is not None:
            self.resume_model()

    def load_spatial_model_separately(self):
        if not os.path.isfile(self.args.separate_spatial_model_path):
            raise RuntimeError(f"=> no checkpoint found at '{self.args.separate_spatial_model_path}'")
        checkpoint = torch.load(self.args.separate_spatial_model_path)

        # spatial_model_dict = self.spatial_model.state_dict()
        # pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in spatial_model_dict}
        # spatial_model_dict.update(pretrained_dict)
        # self.spatial_model.load_state_dict(spatial_model_dict)
        # for param in self.spatial_model.parameters():
        #     param.requires_grad = False

        if self.args.cuda:
            self.spatial_model.module.load_state_dict(checkpoint['state_dict'])
        else:
            self.spatial_model.load_state_dict(checkpoint['state_dict'])

        self.best_spatial_pred = checkpoint['best_pred']

        if self.args.print_ft:
            print(f"=> loaded spatial checkpoint '{self.args.separate_spatial_model_path}' (best pred {self.best_spatial_pred})")

    def resume_model(self):
        if not os.path.isfile(self.args.resume):
            raise RuntimeError(f"=> no checkpoint found at '{self.args.resume}'")
        checkpoint = torch.load(self.args.resume)
        self.args.start_epoch = checkpoint['epoch']

        # Model + Discriminator
        if self.args.cuda:
            self.spatial_model.module.load_state_dict(checkpoint['spatial_model_state_dict'])
            self.temporal_model.module.load_state_dict(checkpoint['temporal_model_state_dict'])
            if self.args.adversarial_loss and 'discriminator_state_dict' in checkpoint:
                if checkpoint['discriminator_state_dict'] is not None:
                    self.discriminator.module.load_state_dict(checkpoint['discriminator_state_dict'])
        else:
            self.spatial_model.load_state_dict(checkpoint['spatial_model_state_dict'])
            self.temporal_model.load_state_dict(checkpoint['temporal_model_state_dict'])
            if self.args.adversarial_loss and 'discriminator_state_dict' in checkpoint:
                if checkpoint['discriminator_state_dict'] is not None:
                    self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

        # Optimizers
        if self.args.ft:
            self.args.start_epoch = 0
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.args.adversarial_loss and 'discriminator_optimizer_state_dict' in checkpoint:
                if checkpoint['discriminator_optimizer_state_dict'] is not None:
                    self.optimizer_D.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
        self.best_pred = checkpoint['best_pred']

        if self.args.print_ft:
            print(f"=> loaded checkpoint '{self.args.resume}' (epoch {checkpoint['epoch']})")

    def model_on_cuda(self):
        self.spatial_model = torch.nn.DataParallel(self.spatial_model, device_ids=self.args.gpu_ids)
        patch_replication_callback(self.spatial_model)
        self.spatial_model = self.spatial_model.cuda()

        self.temporal_model = torch.nn.DataParallel(self.temporal_model, device_ids=self.args.gpu_ids)
        patch_replication_callback(self.temporal_model)
        self.temporal_model = self.temporal_model.cuda()

        if self.args.adversarial_loss:
            self.discriminator = torch.nn.DataParallel(self.discriminator, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.discriminator)
            self.discriminator = self.discriminator.cuda()

    def define_optimizer(self):
        if self.args.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.temporal_model.parameters(),
                                             lr=self.args.lr,
                                             momentum=self.args.momentum,
                                             weight_decay=self.args.weight_decay,
                                             nesterov=self.args.nesterov)
        elif self.args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.temporal_model.parameters(),
                                              lr=self.args.lr,
                                              weight_decay=self.args.weight_decay)
        else:
            raise RuntimeError(f"Chosen optimizer is not usable !!!")

    def training(self, epoch):
        train_loss = 0.0
        self.temporal_model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)

        for i, sample in enumerate(tbar):
            image, target, random_image = sample['image'], sample['label'], sample['random_image']
            if self.args.cuda:
                image, target, random_image = image.cuda(), target.cuda(), random_image.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output, dev_estimate, target_dev = self.temporal_model(image, random_input=random_image, train = True)
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
                self.summary.visualize_image_temporal(self.writer, self.args.dataset, image, target, random_image,
                                                      output, global_step)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        if self.args.no_val:
            # save checkpoint every epoch
            self.save(epoch, is_best=False)

    def validation(self, epoch):
        self.temporal_model.eval()
        self.evaluator.reset()

        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target, random_image = sample['image'], sample['label'], sample['random_image']
            if self.args.cuda:
                image, target, random_image = image.cuda(), target.cuda(), random_image.cuda()
            with torch.no_grad():
                output, dev_estimate, target_dev = self.temporal_model(image, random_input=random_image, train=True)
            loss = self.criterion(output, target)
            test_loss += loss.item()

            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))

            if i == 0 :
                self.summary.visualize_image_temporal(self.writer, self.args.dataset, image, target, random_image,
                                                      output, epoch, name='_val')

            target = target.cpu().numpy()
            pred = np.argmax(output.data.cpu().numpy(), axis=1)

            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        new_pred = self.measure_performance(self.evaluator, epoch, test_loss, image, i)

        if new_pred > self.best_pred:
            self.best_pred = new_pred
            self.save(epoch, is_best=True)

    def save(self, epoch, is_best):
        self.saver.save_checkpoint({
            'epoch': epoch + 1,
            'spatial_model_state_dict': self.spatial_model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
            'discriminator_state_dict': self.discriminator.module.state_dict() if self.discriminator is not None else None,
            'discriminator_optimizer_state_dict': self.optimizer_D.state_dict() if self.optimizer_D is not None else None,
            'temporal_model_state_dict': self.temporal_model.module.state_dict()
        }, is_best)


def main():
    args = get_args()
    print(args)
    torch.manual_seed(args.seed)
    trainer = TemporalTrainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    main()
