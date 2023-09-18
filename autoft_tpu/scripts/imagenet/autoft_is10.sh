#!/bin/bash
source /iris/u/cchoi1/robust-optimizer/ropt/bin/activate

cd ../..

export PYTHONPATH="${PYTHONPATH}:/iris/u/cchoi1/robust-optimizer/autoft/"

python3 src/finetune3.py --method autoft --model ViT-B/16 \
--data-location /home/carolinechoi/robust-ft \
--id ImageNet --ood ImageNetC --eval-datasets ImageNetA,ImageNetR,ImageNetSketch,ImageNetV2,ObjectNet,ImageNet,ImageNetC \
--num_ood_hp_examples 30000 \
--ft_epochs 10 --inner_steps 10 --autoft_epochs 1000 \
--lr 3e-5 --wd 0.1 --batch-size 16 --warmup_length 1000 --accumulation_steps 4 \
--load /home/carolinechoi/robust-ft/zeroshot/clip_vitb16_imagenet.pt \
--load_hparams /home/carolinechoi/robust-ft/hparams/ImageNet/is=10_evals=1000_ex=15000.json