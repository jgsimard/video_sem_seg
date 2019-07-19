#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2
python train.py --backbone resnet \
                --lr 0.0001 \
                --workers 4 \
                --epochs 50 \
                --batch-size 16 \
                --gpu-ids 0,1,2\
                --eval-interval 1 \
                --dataset isi_intensity \
                --epochs 300 \
                --dataset_dir /home/deepsight/data/sem_seg_multiview_07_10_2019 \
                --workers 32 \
                --ft \
                --resume ./run/isi_intensity/deeplab-resnet/model_best.pth.tar \
                --out-stride 8 \
                --loss_type dice \
#                --GaussCrf \
#                --TrainCrf
