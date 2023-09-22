#!/bin/bash
source /home/carolinechoi/robust-optimizer/ropt/bin/activate

cd ../..

export PYTHONPATH="${PYTHONPATH}:/home/carolinechoi/robust-optimizer/autoft/"

python3 src/main.py --method autoft --model ViT-L/14 \
--id CIFAR10 --ood CIFAR10C --eval-datasets CIFAR101,CIFAR102,CIFAR10,CIFAR10C \
--num_ood_hp_examples 100 --ft_epochs 10 \
--autoft_epochs 100 --inner_steps 10 --lr 3e-5 --wd 0.1 --batch-size 64 --accumulation_steps 4 --warmup_length 1000 \
--data-location /home/carolinechoi/robust-ft \
--load /home/carolinechoi/robust-ft/zeroshot/clip_vitl14_openai_cifar10.pt

python3 src/main.py --method autoft --model ViT-L/14 \
--id CIFAR10 --ood CIFAR10C --eval-datasets CIFAR101,CIFAR102,CIFAR10,CIFAR10C \
--num_ood_hp_examples 100 --ft_epochs 10 \
--autoft_epochs 100 --inner_steps 10 --lr 3e-5 --wd 0.1 --batch-size 64 --accumulation_steps 4 --warmup_length 1000 \
--data-location /home/carolinechoi/robust-ft \
--load /home/carolinechoi/robust-ft/zeroshot/clip_vitl14_openai_cifar10.pt \
--load_hparams /home/carolinechoi/robust-ft/hparams/CIFAR10/is=10_evals=100_ex=100_idval.json