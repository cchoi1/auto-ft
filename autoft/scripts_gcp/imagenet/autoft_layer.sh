#!/bin/bash
source /home/carolinechoi/robust-optimizer/ropt/bin/activate

cd ../..

export PYTHONPATH="${PYTHONPATH}:/home/carolinechoi/robust-optimizer/autoft/"

python3 src/main.py --method autoft --model ViT-B/16 --loss_type LayerwiseLoss \
--id ImageNet --ood ImageNetC --eval-datasets ImageNetA,ImageNetR,ImageNetSketch,ImageNetV2,ObjectNet,ImageNet,ImageNetC \
--ft_epochs 10 --inner_steps 10 --autoft_epochs 10000 \
--lr 3e-5 --wd 0.1 --batch-size 512 --warmup_length 1000 \
--data-location /home/carolinechoi/robust-ft \
--load /home/carolinechoi/robust-ft/zeroshot/clip_vitb16_imagenet.pt \
--num_ood_hp_examples 100