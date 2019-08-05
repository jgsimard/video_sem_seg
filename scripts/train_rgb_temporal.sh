#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=3
DATASET_DIR=/home/deepsight/data/rgb

python train_temporal.py --backbone xception \
                --lr 0.0001 \
                --workers 8 \
                --batch-size 16 \
                --cuda_visible_devices 2 \
                --gpu-ids 0,1 \
                --checkname deeplab-xception-flow-adv \
                --eval-interval 1 \
                --dataset isi_rgb_temporal \
                --epochs 400 \
                --dataset_dir $DATASET_DIR \
                --optimizer Adam \
                --skip_classes cannula,instrument \
                --img_shape 513,513 \
                --separate_spatial_model_path ./run/isi_rgb/deeplab-xception-adv/experiment_4/checkpoint.pth.tar \
                --svc_kernel_size 11 \
                --train_distance 3000 \
                --flow \
                --adversarial_loss \
                --lr_ratio 1.0 \
                --n_critic 1 \
                --generator_loss_weight 0.0005
