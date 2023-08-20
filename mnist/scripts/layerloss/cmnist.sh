#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --account=iris # Run on IRIS nodes
#SBATCH --exclude=iris4,iris7
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=8G # Request 16GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name="layerloss-coloredmnist" # Name the job (for easier monitoring)
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

# Now your Python or general experiment/job runner code
source /iris/u/cchoi1/robust-optimizer/ropt/bin/activate
cd ../..

python3 main.py --method ours --ft_dists id --pretrain svhn --id mnist --ood mnistc --test colored_mnist \
--ood_samples_per_class 50 --output_channels 3 --val ood \
--loss_name LayerLoss \
--inner_lr 1e-1 --inner_steps 2 \
--meta_steps 500 --meta_lr 1e-1 --meta_batch_size 20 \
--val_inner_steps 50 --val_meta_batch_size 1 \
--seeds 0 --no_wandb

python3 main.py --method ours --ft_dists id --pretrain svhn --id mnist --ood mnistc --test colored_mnist \
--id_samples_per_class 50 --ood_samples_per_class 50 --output_channels 3 --val ood \
--loss_name LayerLoss \
--inner_lr 1e-1 --inner_steps 100 \
--meta_steps 300 --meta_lr 1e-1 --meta_batch_size 20 \
--val_inner_steps 50 --val_meta_batch_size 1 \
--seeds 0 --no_wandb