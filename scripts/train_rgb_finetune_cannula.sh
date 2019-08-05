#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2
DATASET_DIR=/home/deepsight/data/rgb_canulasse
#DATASET_DIR=/home/deepsight/data/rgb_corrected
#DATASET_DIR = /home/deepsight2/development/data/rgb

python train.py --backbone xception \
                --lr 0.0001 \
                --workers 8 \
                --batch-size 12 \
                --cuda_visible_devices 2 \
                --gpu-ids 0 \
                --checkname deeplab-xception-cannula-pretrained \
                --eval-interval 1 \
                --dataset isi_rgb \
                --epochs 400 \
                --dataset_dir $DATASET_DIR \
                --optimizer Adam \
                --adversarial_loss \
                --lr_ratio 0.8 \
                --n_critic 1 \
                --generator_loss_weight 0.001 \
                --skip_classes instrument \
                --img_shape 513,513 \
                --resume ./run/isi_rgb/deeplab-xception-adv/experiment_4/checkpoint.pth.tar \
                --ft

