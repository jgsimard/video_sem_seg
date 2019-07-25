#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1;
python train.py --backbone resnet \
                --lr 0.0007 \
                --workers 4 \
                --batch-size 16 \
                --gpu-ids 0\
                --checkname deeplab-resnet-unbalenced-adv \
                --eval-interval 1 \
                --dataset isi_rgb \
                --epochs 200 \
                --dataset_dir /home/deepsight2/development/data/rgb \
                --optimizer Adam \
                --adversarial_loss

#                --out-stride 8 \
#                --GaussCrf \
#                --TrainCrf
#                --ft \
#                --resume ./run/isi/deeplab-resnet/model_best.pth.tar \
#                --GaussCrf \
#                --TrainCrf \
#                --freeze-bn True
