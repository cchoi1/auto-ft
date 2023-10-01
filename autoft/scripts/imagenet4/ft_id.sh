#!/bin/bash
python3 src/main.py --method ft-id --model ViT-B/16 --data-location /iris/u/yoonho/data \
--id ImageNet4 --ood ImageNetC --eval-datasets ImageNetA,ImageNetR,ImageNetSketch,ImageNetV2,ObjectNet,ImageNet4,ImageNetC \
--ft_epochs 11 --inner_steps 10 --autoft_epochs 100 --val_freq 10 \
--lr 1e-5 --wd 0.1 --batch-size 512 --warmup_length 500 --accumulation_steps 1 \
--load ./zeroshot/clip_vitb16_imagenet.pt \
--num_ood_hp_examples 10000