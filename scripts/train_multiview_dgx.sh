#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1
python train_multiview.py --backbone resnet \
                --lr 0.0001 \
                --workers 8 \
                --epochs 200 \
                --batch-size 4 \
                --gpu-ids 0\
                --checkname deeplab-multiview-merger \
                --eval-interval 1 \
                --dataset isi_multiview \
                --loss_type ce \
                --dataset_dir /home/deepsight/data/sem_seg_multiview_07_10_2019 \
                --path_pretrained_model /home/deepsight/video_sem_seg/run/isi_intensity/deeplab-resnet/experiment_2/checkpoint.pth.tar \
                --lr_scheduler poly\
                --optimizer Adam \
                --adversarial_loss True \

#                --resume /home/deepsight/multiview/video_sem_seg_multiview/run/isi_multiview/deeplab-multiview-merger/experiment_3/checkpoint.pth.tar \
#                --GaussCrf
#                --ft \
#                --resume ./deeplab-resnet_imagenet.pth.tar