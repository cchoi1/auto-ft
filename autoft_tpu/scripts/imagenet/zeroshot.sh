#!/bin/bash
source /iris/u/cchoi1/robust-optimizer/ropt/bin/activate

cd ../..

export PYTHONPATH="${PYTHONPATH}:/iris/u/cchoi1/robust-optimizer/autoft_tpu/"

python3 src/models/zeroshot.py --method autoft --model ViT-B/16 \
--data-location /home/carolinechoi/data \
--id ImageNet --ood ImageNetC --eval-datasets ImageNetA,ImageNetR,ImageNetSketch,ImageNetV2,ObjectNet,ImageNet \
--num_ood_hp_examples 30000 \
--ft_epochs 10 --inner_steps 10 --autoft_epochs 1000 \
--lr 1e-5 --wd 0.1 --batch-size 16 --warmup_length 500 --accumulation_steps 4 \
--template openai_imagenet_template --save /home/carolinechoi/data/zeroshot/clip_vitb16_imagenet.pt \
--losses ce dcm entropy flyp hinge l1init l1zero l2init l2zero