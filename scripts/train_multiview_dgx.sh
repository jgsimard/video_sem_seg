#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1
python train_multiview.py --backbone xception \
                --lr 0.0001 \
                --workers 16 \
                --epochs 50 \
                --batch-size 4 \
                --gpu-ids 0\
                --eval-interval 1 \
                --loss-type ce \
                --optimizer Adam \
                --adversarial_loss True \
                --n_critic 5\
                --generator_loss_weight 0.003 \
                --unet_size Small \
                --path_pretrained_model /home/deepsight/video_sem_seg/run/isi_intensity/multiview-spatial-xception-adv/experiment_0/checkpoint.pth.tar \
                --checkname deeplab-multiview-inputaug-randomdepth \
                --dataset isi_multiview \
                --dataset_dir /home/deepsight/data/sem_seg_multiview_07_10_2019 \
                --skip_classes cielinglight,floor,table,chair,wall,anesthesiacart,cannula \

#                --resume /home/deepsight/multiview/video_sem_seg_multiview/run/isi_multiview/deeplab-multiview-inputaug/experiment_9/checkpoint.pth.tar \
#                --validation_only \

#                --resume /home/deepsight/multiview/video_sem_seg_multiview/run/isi_multiview/deeplab-multiview-inputaug/experiment_5/checkpoint.pth.tar \
#                --GaussCrf \
#                --ft \
#                --resume ./deeplab-resnet_imagenet.pth.tar \
#                --skip_classes cielinglight,floor,table,chair,wall,anesthesiacart,cannula \ 2019
#                --skip_classes cielinglight,wall \ 2018
#                --separable_conv \

# classes for 2018 data: ORTable/Robot/Human/CielingLight/Floor/StandTable/Chair/Wall/VisionCart
