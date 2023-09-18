#!/bin/bash
source /home/carolinechoi/robust-optimizer/ropt/bin/activate

cd ../..

export PYTHONPATH="${PYTHONPATH}:/home/carolinechoi/robust-optimizer/autoft/"

python3 src/main.py --method ft-id --model ViT-L/14 \
--id CIFAR10 --ood CIFAR10C --eval-datasets CIFAR101,CIFAR102,CIFAR10,CIFAR10C \
--lr 3e-5 --wd 0.1 --batch-size 256 --warmup_length 1000 \
--num_ood_hp_examples 190 --ft_epochs 10 \
--data-location /home/carolinechoi/robust-ft/ \
--load /home/carolinechoi/robust-ft/zeroshot/clip_vitl14_openai_cifar10.pt