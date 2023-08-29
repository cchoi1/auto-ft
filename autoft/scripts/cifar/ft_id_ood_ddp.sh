#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --account=iris # Run on IRIS nodes
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris7
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=16G # Request 16GB of memory
#SBATCH --gres=gpu:2 # Request one GPU
#SBATCH --job-name="ddp-cifar-ft-id-ood" # Name the job (for easier monitoring)
#SBATCH --output=cifar-ft-id-ood-ddp.log  # Name of the output log file
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

source /iris/u/cchoi1/robust-optimizer/ropt/bin/activate

cd ../..

export PYTHONPATH="${PYTHONPATH}:/iris/u/cchoi1/robust-optimizer/autoft/"

python3 -m torch.distributed.launch --nproc_per_node=2 \
src/main.py --distributed --method ft-id-ood --model ViT-L/14 --data-location /iris/u/cchoi1/Data \
--id CIFAR10 --ood CIFAR10C --eval-datasets CIFAR101,CIFAR102,CIFAR10,CIFAR10C \
--lr 7.5e-6 --wd 0.1 --batch-size 128 --warmup_length 2000 --workers 4 \
--load /iris/u/cchoi1/robust-optimizer/autoft/zeroshot/clip_vitl14_openai_cifar10.pt --save ft-id-ood-ddp- --exp_name CIFAR10 \
--num_ood_examples 10000 --num_ood_hp_examples 100 --ft_epochs 10 --results-db ./results/CIFAR/ft-id-ood-ddp