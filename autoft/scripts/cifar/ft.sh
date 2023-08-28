#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --account=iris # Run on IRIS nodes
#SBATCH --exclude=iris7
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=16G # Request 16GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name="cifar-ft-id-ood" # Name the job (for easier monitoring)
#SBATCH --output=cifar-ft-id-ood.log  # Name of the output log file
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

source /iris/u/cchoi1/robust-optimizer/ropt/bin/activate

cd ../..

export PYTHONPATH="${PYTHONPATH}:/iris/u/cchoi1/robust-optimizer/autoft/"

for NUM_OOD_EXAMPLES in 100 1000 10000
do
  for INNER_STEPS in 10 100 1000
  do
    RUN_NAME="cifar_ft_no=${NUM_OOD_EXAMPLES}_is=${INNER_STEPS}"
    python src/main.py --method ft-id-ood --model RN50 \
    --data-location /iris/u/yoonho/data --id CIFAR10 --ood CIFAR10C --eval-datasets CIFAR101 \
    --num_ood_examples 10000 --num_ood_hp_examples $NUM_OOD_EXAMPLES --inner_steps $INNER_STEPS \
    --lr 1e-4 --batch-size 32 --workers 2 --load ./zeroshot/clip_rn50_simple.pt \
    --save ft-id-ood- --exp_name $RUN_NAME --results-db $RUN_NAME
  done
done