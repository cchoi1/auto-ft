#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --account=iris # Run on IRIS nodes
#SBATCH --exclude=iris7
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=16G # Request 16GB of memory
#SBATCH --gres=gpu:2 # Request one GPU
#SBATCH --job-name="cifar-autoft" # Name the job (for easier monitoring)
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

cd ../..

export PYTHONPATH="${PYTHONPATH}:/iris/u/cchoi1/robust-optimizer/autoft/"

python src/main.py --method autoft --model RN50 \
--data-location /iris/u/yoonho/data --id CIFAR10 --ood CIFAR10C --eval-datasets CIFAR101 \
--num_ood_examples 10000 --num_ood_hp_examples 100 --inner_steps 100 --val_freq 10 --epochs 100 \
--lr 1e-4 --batch-size 64 --workers 2 --load ./zeroshot/clip_rn50_simple.pt \
--save autoft- --exp_name CIFAR10