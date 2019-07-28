#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=3
DATASET_DIR=/home/deepsight/data/rgb

python train_temporal.py --backbone xception \
                --lr 0.0001 \
                --workers 8 \
                --batch-size 6 \
                --cuda_visible_devices 2 \
                --gpu-ids 0 \
                --checkname deeplab-xception \
                --eval-interval 1 \
                --dataset isi_rgb_temporal \
                --epochs 400 \
                --dataset_dir $DATASET_DIR \
                --optimizer Adam \
                --adversarial_loss \
                --lr_ratio 0.8 \
                --n_critic 1 \
                --generator_loss_weight 0.001 \
                --skip_classes cannula,instrument \
                --img_shape 513,513 \
                --separate_spatial_model_path ./run/isi_rgb/deeplab-xception-adv/experiment_4/checkpoint.pth.tar
