#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --account=iris # Run on IRIS nodes
#SBATCH --exclude=iris4
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=8G # Request 16GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name="full_l2-sp-svhn" # Name the job (for easier monitoring)
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

# Now your Python or general experiment/job runner code
source /iris/u/cchoi1/robust-optimizer/ropt/bin/activate
cd ../

echo "VAL OOD"
for LAMBDA in 0.001 0.01 0.1 0.5 1.0 2.0 5.0 10.0
do
    echo "L2 LAMBDA: $LAMBDA"
#    python3 main.py --method full --ft_id_dist brightness --ft_ood_dist impulse_noise --test_dist mnistc \
#    --l2_lambda $LAMBDA --patience 3 --val ood
    python3 main.py --method full --pretrain_dist svhn --ft_id_dist mnist --ft_ood_dist impulse_noise --test_dist mnistc \
    --optimizer_name LayerSGD --l2_lambda $LAMBDA --patience 3 --val ood --no_wandb
done