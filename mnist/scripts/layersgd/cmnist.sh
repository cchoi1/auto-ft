#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --account=iris # Run on IRIS nodes
#SBATCH --exclude=iris4,iris7
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=8G # Request 16GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name="layer-sgd-cmnist" # Name the job (for easier monitoring)
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

# Now your Python or general experiment/job runner code
source /iris/u/cchoi1/robust-optimizer/ropt/bin/activate
cd ../..

#python3 main.py --method ours --pretrain svhn --id mnist --ood mnistc --test colored_mnist --output_channels 3 \
#--id_samples_per_class 50 --ood_samples_per_class 50 \
#--optimizer_name LayerSGD --ft_dists id --val ood \
#--inner_steps 5 --meta_steps 150 --seeds 0

python3 main.py --method ours --pretrain svhn --id mnist --ood mnistc --test colored_mnist --output_channels 3 \
--id_samples_per_class 50 --ood_samples_per_class 50 \
--optimizer_name LayerSGD --momentum --ft_dists id --val ood \
--inner_steps 5 --meta_steps 150 --seeds 0

python3 main.py --method ours --pretrain svhn --id mnist --ood mnistc --test colored_mnist --output_channels 3 \
--id_samples_per_class 50 --ood_samples_per_class 50 \
--optimizer_name LayerSGD --wnb --ft_dists id --val ood \
--inner_steps 5 --meta_steps 150 --seeds 0

python3 main.py --method ours --pretrain svhn --id mnist --ood mnistc --test colored_mnist --output_channels 3 \
--id_samples_per_class 50 --ood_samples_per_class 50 \
--optimizer_name LayerSGD --wnb --momentum --ft_dists id --val ood \
--inner_steps 5 --meta_steps 150 --seeds 0