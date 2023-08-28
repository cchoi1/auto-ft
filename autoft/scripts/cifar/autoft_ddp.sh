#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --account=iris # Run on IRIS nodes
#SBATCH --exclude=iris7
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=16G # Request 16GB of memory
#SBATCH --gres=gpu:4 # Request one GPU
#SBATCH --job-name="cifar-autoft-ddp" # Name the job (for easier monitoring)
#SBATCH --output=cifar-autoft-ddp.log  # Name of the output log file
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

source /iris/u/cchoi1/robust-optimizer/ropt/bin/activate

cd ../..

export PYTHONPATH="${PYTHONPATH}:/iris/u/cchoi1/robust-optimizer/autoft/"

NUM_OOD_EXAMPLES=100
INNER_STEPS=100
EPOCHS=100
RUN_NAME=cifar_autoft_no=$NUM_OOD_EXAMPLES_is=$INNER_STEPS_ep=$EPOCHS

python3 -m torch.distributed.launch --nproc_per_node=4 \
src/main.py --distributed --method autoft --model RN50 \
--data-location /iris/u/yoonho/data --id CIFAR10 --ood CIFAR10C --eval-datasets CIFAR101 \
--num_ood_examples 10000 --num_ood_hp_examples $NUM_OOD_EXAMPLES --inner_steps $INNER_STEPS --val_freq 10 --epochs $EPOCHS \
--lr 1e-4 --batch-size 64 --workers 2 --load ./zeroshot/clip_rn50_simple.pt \
--save autoft- --exp_name RUN_NAME --results_db RUN_NAME

