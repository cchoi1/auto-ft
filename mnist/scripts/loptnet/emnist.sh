#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --account=iris # Run on IRIS nodes
#SBATCH --exclude=iris4,iris7
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=8G # Request 16GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name="loptnet-emnist" # Name the job (for easier monitoring)
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

# Now your Python or general experiment/job runner code
source /iris/u/cchoi1/robust-optimizer/ropt/bin/activate
cd ../..


#python3 main.py --method ours --pretrain_dist svhn --ft_id_dist mnist --ft_ood_dist mnistc --test_dist emnist --output_channels 3 \
#--id_samples_per_class 50 --ood_samples_per_class 50 \
#--optimizer_name LOptNet --output lr_multiplier --features depth --lopt_net_dim 4 \
#--inner_steps 10 --meta_steps 150 --val ood --seeds 0
#
#python3 main.py --method ours --pretrain_dist svhn --ft_id_dist mnist --ft_ood_dist mnistc --test_dist emnist --output_channels 3 \
#--id_samples_per_class 50 --ood_samples_per_class 50 \
#--optimizer_name LOptNet --output lr_multiplier --wnb --features depth --lopt_net_dim 4 \
#--inner_steps 10 --meta_steps 150 --val ood --seeds 0

python3 main.py --method ours --pretrain_dist svhn --ft_id_dist mnist --ft_ood_dist mnistc --test_dist emnist --output_channels 3 \
--id_samples_per_class 50 --ood_samples_per_class 50 \
--optimizer_name LOptNet --output lr_multiplier --momentum --features g depth --lopt_net_dim 4 \
--inner_steps 5 --meta_steps 150 --val ood --seeds 0

#python3 main.py --method ours --pretrain_dist svhn --ft_id_dist mnist --ft_ood_dist mnistc --test_dist emnist --output_channels 3 \
#--id_samples_per_class 50 --ood_samples_per_class 50 \
#--optimizer_name LOptNet --output lr_multiplier --wnb --momentum --features depth --lopt_net_dim 4 \
#--inner_steps 10 --meta_steps 150 --val ood --seeds 0