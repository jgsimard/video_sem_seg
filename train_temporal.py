import argparse
import os
import numpy as np
from tqdm import tqdm

import torch

from mypath import Path
from datasets import make_data_loader
from models.modeling.sync_batchnorm.replicate import patch_replication_callback
# from models.modeling.deeplab import *
from models.modeling.deeplab import DeepLab
from models.afp import LowLatencyModel
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator

from train import Trainer, seed, get_args
import models.convcrf as convcrf

class TemporalTrainer(Trainer):
    def __init__(self, args):
        Trainer.__init__(self, args=args)
        self.temporal_model = LowLatencyModel(self.model)

        # Fix deeplab as the features extractor
        for param in self.temporal_model.deeplab.parameters():
            param.requires_grad = False

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
            output = self.temporal_model.forward_train(image, random_image)
            print(random_image.shape)
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
                self.summary.visualize_image_temporal(self.writer, self.args.dataset, image, target, random_image, output, global_step)

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
                output = self.temporal_model.forward_train(image, random_image)
            loss = self.criterion(output, target)
            test_loss += loss.item()

            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))

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
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
            'crf_state_dict': self.crf.module.state_dict() if self.crf is not None else None,
            'discriminator_state_dict': self.discriminator.module.state_dict() if self.discriminator is not None else None,
            'discriminator_optimizer_state_dict' : self.optimizer_D.state_dict() if self.optimizer_D is not None else None,
            'temporal_model_state_dict': self.temporal_model.state_dict()
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
