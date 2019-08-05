#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1
python train_multiview.py --backbone xception \
                --lr 0.0001 \
                --workers 8 \
                --epochs 50 \
                --batch-size 4 \
                --gpu-ids 0\
                --checkname deeplab-multiview-inputaug \
                --eval-interval 1 \
                --dataset isi_multiview \
                --loss_type ce \
                --dataset_dir /home/deepsight/data/sem_seg_multiview_07_10_2019 \
                --path_pretrained_model /home/deepsight/video_sem_seg/run/isi_intensity/multiview-spatial-xception-adv/experiment_0/checkpoint.pth.tar \
                --optimizer Adam \
                --adversarial_loss True \
                --generator_loss_weight 0.0005 \
                --unet_size Small \
                --resume /home/deepsight/multiview/video_sem_seg_multiview/run/isi_multiview/deeplab-multiview-inputaug/experiment_5/checkpoint.pth.tar \
#                --GaussCrf \
#                --ft \
#                --resume ./deeplab-resnet_imagenet.pth.tar