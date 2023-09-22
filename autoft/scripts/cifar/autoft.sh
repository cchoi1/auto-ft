#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --account=iris # Run on IRIS nodes
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris7
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=16G # Request 16GB of memory
#SBATCH --gres=gpu:2 # Request one GPU
#SBATCH --job-name="cifar-autoft-10inner-500ep-190" # Name the job (for easier monitoring)
#SBATCH --output=cifar-autoft-10inner-500ep-190.log  # Name of the output log file
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

source /iris/u/cchoi1/robust-optimizer/ropt/bin/activate

cd ../..

export PYTHONPATH="${PYTHONPATH}:/iris/u/cchoi1/robust-optimizer/autoft/"

#python3 src/main.py --method autoft --model ViT-L/14 --data-location /iris/u/cchoi1/Data \
#--id CIFAR10 --ood CIFAR10C --eval-datasets CIFAR101,CIFAR102,CIFAR10,CIFAR10C \
#--num_ood_hp_examples 190 --ft_epochs 10 \
#--autoft_epochs 100 --inner_steps 1 --lr 3e-5 --wd 0.1 --batch-size 64 --warmup_length 4000 --workers 4 \
#--load /iris/u/cchoi1/robust-optimizer/autoft/zeroshot/clip_vitl14_openai_cifar10.pt

python3 src/main.py --method autoft --model ViT-L/14 --data-location /iris/u/cchoi1/Data \
--id CIFAR10 --ood CIFAR10C --eval-datasets CIFAR101,CIFAR102,CIFAR10,CIFAR10C \
--num_ood_hp_examples 100 --ft_epochs 10 \
--autoft_epochs 100 --inner_steps 10 --lr 3e-5 --wd 0.1 --batch-size 64 --warmup_length 1000 --workers 4 \
--load /iris/u/cchoi1/robust-optimizer/autoft/zeroshot/clip_vitl14_openai_cifar10.pt

# python3 src/main.py --method autoft --model ViT-L/14 --data-location /iris/u/cchoi1/Data --id CIFAR10 --ood CIFAR10C --eval-datasets CIFAR101,CIFAR102,CIFAR10,CIFAR10C --autoft_epochs 100 --val_freq 10 --inner_steps 1 --lr 1.875e-6 --wd 0.1 --batch-size 32 --warmup_length 8000 --workers 0 --load /iris/u/cchoi1/robust-optimizer/autoft/zeroshot/clip_vitl14_openai_cifar10.pt --save autoft- --exp_name CIFAR10 --num_ood_examples 10000 --num_ood_hp_examples 100 --ft_epochs 10 --results-db ./results/CIFAR/autoft