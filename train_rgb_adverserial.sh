#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2;
python train.py --backbone resnet \
                --lr 0.0007 \
                --workers 32 \
                --batch-size 16 \
                --gpu-ids 0\
                --checkname deeplab-resnet-adv-unbalanced \
                --eval-interval 1 \
                --dataset isi_rgb \
                --epochs 200 \
                --dataset_dir /home/deepsight/data/rgb \
                --use-balanced-weights \
                --adversarial_loss True
#                --out-stride 8 \
#                --GaussCrf \
#                --TrainCrf
#                --ft \
#                --resume ./run/isi/deeplab-resnet/model_best.pth.tar \
#                --GaussCrf \
#                --TrainCrf \
#                --freeze-bn True
