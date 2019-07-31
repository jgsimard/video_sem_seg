#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1;
#DATASET_DIR=/home/deepsight/data/rgb_canulasse
DATASET_DIR=/home/deepsight/data/rgb
CHECKNAME=spatial-xception-adv-resume

python train.py --backbone xception \
                --lr 0.0001 \
                --workers 32 \
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
                --n_critic 1 \
                --generator_loss_weight 0.001 \
                --img_shape 513,513 \
                --skip_classes cannula,instrument \
                --resume ./scripts/run/isi_rgb/spatial-xception-adv/experiment_2/checkpoint.pth.tar
#                --loss-type focal

