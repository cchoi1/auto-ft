#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --account=iris # Run on IRIS nodes
#SBATCH --exclude=iris4,iris7
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=8G # Request 16GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name="ft-id-then-ood-l2-sp" # Name the job (for easier monitoring)
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

# Now your Python or general experiment/job runner code
source /iris/u/cchoi1/robust-optimizer/ropt/bin/activate
cd ../..

for l2_lambda in 0.01 0.1 0.5 1.0 2.0 5.0 10.0
do
  echo "val id"
  python3 main.py --method full full --ft_dists id ood --ft_lr 5e-2 \
  --pretrain svhn --id mnist --ood mnistc --test colored_mnist --ood_samples_per_class 50 \
  --output_channels 3 \
  --val id ood --optimizer_name LayerSGD --l2_lambda $l2_lambda \
  --seeds 0 --no_wandb
done
