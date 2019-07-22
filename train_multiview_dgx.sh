#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0
python train_multiview.py --backbone resnet \
                --lr 0.00005 \
                --workers 8 \
                --batch-size 4 \
                --gpu-ids 0\
                --checkname deeplab-multiview-merger \
                --eval-interval 1 \
                --dataset isi_multiview \
                --epochs 100 \
                --loss_type ce \
                --dataset_dir /home/deepsight/data/sem_seg_multiview_07_10_2019 \
                --path_pretrained_model /home/deepsight/video_sem_seg/run/isi_intensity/deeplab-resnet/experiment_2/checkpoint.pth.tar \
                --lr_scheduler poly\

#                --resume /home/deepsight3/dev/deepsight/MultiView/script/video_sem_seg/run/isi/deeplab-multiview-relufeature/experiment_0/checkpoint.pth.tar \
#                --GaussCrf
#                --ft \
#                --resume ./deeplab-resnet_imagenet.pth.tar