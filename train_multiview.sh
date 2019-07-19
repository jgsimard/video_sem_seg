#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0
python train_multiview.py --backbone resnet \
                --lr 0.0001 \
                --workers 8 \
                --epochs 50 \
                --batch-size 4 \
                --gpu-ids 0\
                --checkname deeplab-multiview-merger \
                --eval-interval 1 \
                --dataset isi_multiview \
                --loss_type ce \
                --dataset_dir /home/deepsight3/dev/deepsight/MultiView/data \
                --path_pretrained_model /home/deepsight3/dev/deepsight/MultiView/script/video_sem_seg_multiview/model_best_multiview.pth.tar \
                --lr_scheduler poly\

#                --resume /home/deepsight3/dev/deepsight/MultiView/script/video_sem_seg/run/isi/deeplab-multiview-relufeature/experiment_0/checkpoint.pth.tar \
#                --GaussCrf
#                --ft \
#                --resume ./deeplab-resnet_imagenet.pth.tar


