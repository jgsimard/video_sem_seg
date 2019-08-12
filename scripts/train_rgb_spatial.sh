#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=3;
DATASET_DIR=/home/deepsight/data/rgb
CHECKNAME=spatial-xception-adv-CLEAN

python train.py --backbone xception \
                --lr 0.0001 \
                --workers 16 \
                --batch-size 12 \
                --cuda_visible_devices 1 \
                --gpu-ids 0 \
                --eval-interval 1 \
                --dataset isi_rgb \
                --checkname $CHECKNAME \
                --epochs 400 \
                --dataset_dir $DATASET_DIR \
                --optimizer Adam \
                --adversarial_loss \
                --lr_ratio 0.8 \
                --n_critic 2 \
                --generator_loss_weight 0.001 \
                --img_shape 513,513 \
                --skip_classes cannula,instrument