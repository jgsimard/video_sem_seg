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

from train import Trainer, get_args

CAM_NUM = 4
HEIGHT = 287
WIDTH = 352


class MultiViewTrainer(Trainer):
    def __init__(self, args):
        self.args = args
        self.prepare_saver()
        self.prepare_tensorboard()
        self.prepare_dataloader()

        # Define network
        self.model = DeepLabMultiView(num_classes=self.nclass,
                                      backbone=args.backbone,
                                      output_stride=args.out_stride,
                                      sync_bn=args.sync_bn,
                                      freeze_bn=args.freeze_bn,
                                      unet_size=args.unet_size,
                                      separable_conv=args.separable_conv)
        self.model.merger = init_net(self.model.merger, type="kaiming", activation_mode='relu', distribution='normal')

        # load pretrain deeplab model
        if args.path_pretrained_model is not None:
            self.load_pretrained_deeplab()

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

    def load_pretrained_deeplab(self):
        pretrained_dict = torch.load(self.args.path_pretrained_model)['state_dict']
        model_dict = self.model.deeplab.state_dict()
        # filter out uncessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        print('loaded pretrained parameter of ', len(pretrained_dict))

        # overwrite existing entries
        model_dict.update(pretrained_dict)
        # load the new state dict
        self.model.deeplab.load_state_dict(model_dict)
        # fix parameters
        for param in self.model.deeplab.parameters():
            param.requires_grad = False

    def define_evaluators(self):
        if self.args.dataset == "isi_intensity":
            weights = np.array([1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1]) > 0.5
        elif (
                self.args.dataset == "isi_rgb" or self.args.dataset == "isi_rgb_temporal" or self.args.dataset == 'isi_multiview' or self.args.dataset == 'isi_multiview_2018') and self.args.skip_classes is not None:
            weights = self.args.skip_weights > 0.5
        else:
            weights = None
        self.evaluator = Evaluator(self.nclass, weights)

    def train_adverserial_multiview(self, output_merger, target, i, num_img_tr, epoch, loss):
        self.optimizer_D.zero_grad()
        # switch training discriminator and network
        self.train_d = not self.train_d if (i + num_img_tr * epoch) % self.args.n_critic == 0 else self.train_d

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

            # gradient_penality = False
            # if gradient_penality:
            #     gp = compute_gradient_penality(self.discriminator, real_input, fake_input)

            # discriminator
            if self.train_d:
                # discriminator loss
                real_loss = self.adv_loss(real_validity - mean_validity_fake, torch.ones_like(real_validity))
                fake_loss = self.adv_loss(fake_validity - mean_validity_real,
                                          torch.ones_like(fake_validity) * (- 1.0))
                loss_D += real_loss + fake_loss  # + gp

            # train network
            else:
                # discriminator loss
                real_loss = self.adv_loss(real_validity - mean_validity_fake,
                                          torch.ones_like(real_validity) * (- 1.0))
                fake_loss = self.adv_loss(fake_validity - mean_validity_real, torch.ones_like(fake_validity))
                loss_G += real_loss + fake_loss

        if self.train_d:
            # backprop
            loss_D.backward()
            self.optimizer_D.step()
            self.writer.add_scalar('train/discriminator_loss', loss_D.item(), i + num_img_tr * epoch)
        else:
            loss += self.args.generator_loss_weight * loss_G
            # backprop
            loss.backward()
            self.optimizer.step()
            self.writer.add_scalar('train/generator_loss', self.args.generator_loss_weight * loss_G.item(),
                                   i + num_img_tr * epoch)

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()

        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)

        for i, sample in enumerate(tbar):
            image, target, pc = sample['image'], sample['label'], sample['pointcloud']

            if self.args.cuda:
                image, target, pc = image.cuda(), target.cuda(), pc.cuda()

            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()

            # go through network
            output_single_view, output_merger, target = self.model(image, pc, target)

            # pixel loss
            loss = 0.0
            for camid in range(CAM_NUM):
                # pixel loss
                pixel_loss = self.criterion(output_merger[:, camid, :, :, :], target[:, camid, :, :])
                loss += pixel_loss
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # adversarial loss
            if self.args.adversarial_loss:
                self.train_adverserial_multiview(output_merger, target, i, num_img_tr, epoch, loss)
            else:
                loss.backward()
                self.optimizer.step()

            # Show 10 * 3 inference results each epoch
            if i % (num_img_tr // 2) == 0:
                global_step = i + num_img_tr * epoch
                self.summary.visualize_image_mulitview(self.writer, self.args.dataset, image[0, :, :, :, :],
                                                       target[0, :, :, :], output_single_view[0, :, :, :, :],
                                                       output_merger[0, :, :, :, :],
                                                       global_step, name='_Train')

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

        new_pred = self.measure_performance(self.evaluator, epoch, test_loss, image, i)

        self.summary.visualize_image_mulitview(self.writer, self.args.dataset, image[0, :, :, :, :],
                                               target[0, :, :, :], output_single_view[0, :, :, :, :],
                                               output_merger[0, :, :, :, :],
                                               epoch, name='_Validation')
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


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    trainer = MultiViewTrainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)

    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()


if __name__ == "__main__":
    main()
