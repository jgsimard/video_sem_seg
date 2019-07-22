#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3
python train.py --backbone resnet \
                --lr 0.0001 \
                --workers 4 \
                --epochs 50 \
                --batch-size 16 \
                --gpu-ids 0,1,2,3\
                --eval-interval 1 \
                --dataset isi_intensity \
                --epochs 300 \
                --dataset_dir /home/deepsight/data/sem_seg_07_10_2019 \
                --workers 32 \
                --out-stride 8 \
                --GaussCrf \
                --TrainCrf


#                --ft \
#                --resume ./run/isi_intensity/deeplab-resnet/experiment_1/checkpoint.pth.tar \
