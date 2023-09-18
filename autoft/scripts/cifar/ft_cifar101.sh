#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --account=iris # Run on IRIS nodes
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris7
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=16G # Request 16GB of memory
#SBATCH --gres=gpu:2
#SBATCH --job-name="ft-cifar101" # Name the job (for easier monitoring)
#SBATCH --output=ft-cifar101.log  # Name of the output log file
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

source /iris/u/cchoi1/robust-optimizer/ropt/bin/activate

cd ../..

export PYTHONPATH="${PYTHONPATH}:/iris/u/cchoi1/robust-optimizer/autoft/"

python src/main.py --method ft-id --model ViT-L/14 --data-location /iris/u/cchoi1/Data \
--id CIFAR101 --ood CIFAR10C --eval-datasets CIFAR101,CIFAR102,CIFAR10,CIFAR10C \
--lr 3e-5 --wd 0.1 --batch-size 64 --warmup_length 4000 --workers 2 \
--load /iris/u/cchoi1/robust-optimizer/autoft/zeroshot/clip_vitl14_openai_cifar10.pt \
--num_ood_hp_examples 100 --ft_epochs 10

# --load /iris/u/cchoi1/robust-optimizer/autoft/zeroshot/clip_vitl14_openai_cifar10.pt \