#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --account=iris # Run on IRIS nodes
#SBATCH --exclude=iris4,iris7
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=8G # Request 16GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name="lp-ft" # Name the job (for easier monitoring)
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

# Now your Python or general experiment/job runner code
source /iris/u/cchoi1/robust-optimizer/ropt/bin/activate
cd ../..

python3 main.py --method lp-ft \
--ft_dists id --pretrain svhn --id mnist --ood mnistc --test colored_mnist \
--optimizer_name LayerSGD --ft_lr 5e-2 --val ood \
--output_channels 3 --seeds 0

python3 main.py --method lp-ft \
--ft_dists id+ood --pretrain svhn --id mnist --ood mnistc --test colored_mnist \
--optimizer_name LayerSGD --ft_lr 5e-2 --val ood --ood_samples_per_class 50 \
--output_channels 3 --seeds 0