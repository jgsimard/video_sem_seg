#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1
DATASET_DIR=/home/deepsight/data/rgb

python train_temporal.py --backbone xception \
                --lr 0.0001 \
                --workers 8 \
                --batch-size 16 \
                --cuda_visible_devices 2 \
                --gpu-ids 0 \
                --checkname deeplab-xception-adv-CLEAN \
                --eval-interval 1 \
                --dataset isi_rgb_temporal \
                --epochs 400 \
                --dataset_dir $DATASET_DIR \
                --optimizer Adam \
                --skip_classes cannula,instrument \
                --img_shape 513,513 \
                --separate_spatial_model_path ./run/isi_rgb/spatial-xception-adv-CLEAN/experiment_1/checkpoint.pth.tar  \
                --svc_kernel_size 9 \
                --train_distance 2000 \
                --temporal_separable \
                --adversarial_loss \
                --lr_ratio 1.0 \
                --n_critic 2 \
                --generator_loss_weight 0.0005
