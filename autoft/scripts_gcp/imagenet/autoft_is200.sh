#!/bin/bash
cd ../..

python src/main.py --method autoft --model ViT-B/16 --data-location /home/carolinechoi/robust-ft \
--id ImageNet --ood ImageNetC --eval-datasets ImageNetA,ImageNetR,ImageNetSketch,ImageNetV2,ObjectNet,ImageNet,ImageNetC \
--ft_epochs 10 --inner_steps 200 --autoft_epochs 100 --val_freq 10 \
--lr 1e-5 --wd 0.1 --batch-size 512 --warmup_length 1000 --accumulation_steps 1 \
--load /home/carolinechoi/robust-ft/zeroshot/clip_vitb16_imagenet.pt \
--num_ood_hp_examples 10000