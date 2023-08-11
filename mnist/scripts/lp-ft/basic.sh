#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --account=iris # Run on IRIS nodes
#SBATCH --exclude=iris4,iris7
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=8G # Request 16GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name="lp-ft-colored_mnist" # Name the job (for easier monitoring)
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

# Now your Python or general experiment/job runner code
source /iris/u/cchoi1/robust-optimizer/ropt/bin/activate
cd ../..

#python3 main.py --method lp-ft --pretrain_dist svhn --ft_id_dist mnist --ft_ood_dist mnistc --test_dist emnist \
#--id_samples_per_class 50 --ood_samples_per_class 50 --output_channels 3 \
#--val ood --optimizer_name LayerSGD --seeds 0

python3 main.py --method lp-ft --pretrain_dist svhn --ft_id_dist mnist --ft_ood_dist mnistc --test_dist colored_mnist \
--id_samples_per_class 50 --ood_samples_per_class 50 --output_channels 3 \
--val ood --optimizer_name LayerSGD --seeds 0