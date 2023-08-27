#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --account=iris # Run on IRIS nodes
#SBATCH --exclude=iris7
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=32G # Request 16GB of memory
#SBATCH --gres=gpu:4 # Request one GPU
#SBATCH --job-name="clip-cifar10" # Name the job (for easier monitoring)
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

# Now your Python or general experiment/job runner code
source /iris/u/cchoi1/robust-optimizer/ropt/bin/activate
cd ..

python -m torch.distributed.launch --nproc_per_node=4 \
 ddp_main.py --pretrain clip --id cifar10 --ood cifar10c --test cinic10 \
 --lr 1e-4 --batch_size 128 --max_iters 12000 --num_workers 2 --repeats 1