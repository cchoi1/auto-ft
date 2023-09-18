#!/bin/bash
source /home/carolinechoi/robust-optimizer/ropt/bin/activate

cd ../..

export PYTHONPATH="${PYTHONPATH}:/home/carolinechoi/robust-optimizer/autoft/"

python3 src/main.py --method ft-id --model ViT-B/16 \
--id ImageNet --ood ImageNetC --eval-datasets ImageNetA,ImageNetR,ImageNetSketch,ImageNetV2,ObjectNet,ImageNet,ImageNetC \
--ft_epochs 10 --lr 3e-5 --wd 0.1 --batch-size 512 \
--num_ood_hp_examples 15000 --warmup_length 1000 \
--data-location /home/carolinechoi/robust-ft \
--load /home/carolinechoi/robust-ft/zeroshot/clip_vitb16_imagenet.pt