#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0
#DATASET_DIR=/home/deepsight/data/sem_seg_07_10_2019 # intensity OR
DATASET_DIR=/home/deepsight/data/sem_seg_multiview_07_10_2019/ # intensity OR
python train.py --backbone xception \
                --lr 0.0001 \
                --workers 8 \
                --batch-size 8 \
                --cuda_visible_devices 1 \
                --gpu-ids 0 \
                --eval-interval 1 \
                --dataset isi_intensity \
                --checkname multiview-deeplab-xception-adv \
                --epochs 300 \
                --dataset_dir $DATASET_DIR \
                --optimizer Adam \
                --adversarial_loss \
                --lr_ratio 1.0 \
                --n_critic 1 \
                --generator_loss_weight 0.001 \
                --img_shape 287,352

