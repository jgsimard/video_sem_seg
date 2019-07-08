#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3
python train.py --backbone resnet \
                --lr 0.007 \
                --workers 4 \
                --use-sbd \
                --epochs 50 \
                --batch-size 16 \
                --gpu-ids 0,1,2,3\
                --checkname deeplab-resnet \
                --eval-interval 1 \
                --dataset isi \
                --epochs 1000
