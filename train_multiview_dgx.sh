#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=3
python train_multiview.py --backbone xception \
                --lr 0.0001 \
                --workers 8 \
                --epochs 200 \
                --batch-size 4 \
                --gpu-ids 0\
                --checkname deeplab-multiview-aug \
                --eval-interval 1 \
                --dataset isi_multiview \
                --loss_type ce \
                --dataset_dir /home/deepsight/data/sem_seg_multiview_07_10_2019 \
                --path_pretrained_model /home/deepsight/video_sem_seg/run/isi_intensity/multiview-deeplab-xception-adv-pretrained/experiment_2/checkpoint.pth.tar \
                --optimizer Adam \
                --adversarial_loss True \
                --generator_loss_weight 0.0005 \
                
#                --resume /home/deepsight/multiview/video_sem_seg_multiview/run/isi_multiview/deeplab-multiview-aug/experiment_3/checkpoint.pth.tar \
#                --GaussCrf \
#                --ft \
#                --resume ./deeplab-resnet_imagenet.pth.tar