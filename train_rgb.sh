#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2
python train.py --backbone resnet \
                --lr 0.000000000 \
                --workers 16 \
                --use-sbd \
                --epochs 50 \
                --batch-size 16 \
                --gpu-ids 0,1,2\
                --checkname deeplab-resnet \
                --eval-interval 1 \
                --dataset isi_rgb \
                --epochs 100 \
                --dataset_dir /home/deepsight/data/rgb \
                --ft \
                --resume ./run/isi/deeplab-resnet/model_best.pth.tar \
                --GaussCrf \
                --TrainCrf \
                --freeze-bn True
