#!/bin/bash
source /home/carolinechoi/robust-optimizer/ropt/bin/activate

cd ../..

export PYTHONPATH="${PYTHONPATH}:/home/carolinechoi/robust-optimizer/autoft/"

python3 src/main.py --method autoft --loss_type LayerwiseLoss --model ViT-L/14 \
--id CIFAR10 --ood CIFAR10C --eval-datasets CIFAR101,CIFAR102,CIFAR10,CIFAR10C \
--num_ood_hp_examples 190 --ft_epochs 10 \
--autoft_epochs 1000 --inner_steps 10 --lr 3e-5 --wd 0.1 --batch-size 256 --warmup_length 1000 \
--data-location /home/carolinechoi/robust-ft/ \
--load /home/carolinechoi/robust-ft/zeroshot/clip_vitl14_openai_cifar10.pt