#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --account=iris # Run on IRIS nodes
#SBATCH --exclude=iris1,iris2,iris3
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=64G # Request 16GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name="fmow-weight-ensemble-plot" # Name the job (for easier monitoring)
#SBATCH --output=fmow-weight-ensemble-plot.log  # Name of the output log file
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

source /iris/u/cchoi1/robust-optimizer/ropt/bin/activate

cd ../..

export PYTHONPATH="${PYTHONPATH}:/iris/u/cchoi1/robust-optimizer/autoft/"

python3 src/wise_ft.py --method autoft --model ViT-B/16 --data-location /iris/u/yoonho/data/wilds \
--id FMOWTrain --ood FMOWOODVal --eval-datasets FMOWIDTest,FMOWOODTest \
--num_ood_hp_examples 1000 --autoft_epochs 200 --inner_steps 10 --ft_epochs 20 \
--lr 1e-5 --wd 0.1 --batch-size 256 --warmup_length 500 --workers 2 \
--load /iris/u/cchoi1/robust-optimizer/autoft/zeroshot/clip_vitb16_fmow2.pt,/iris/u/cchoi1/robust-optimizer/autoft/saved/FMOWTrain/autoft/oodFMOWOODVal_flyp_cdf/no1000_nouNone_afep100_is10_ftep20_bs128_wd0.1_lr1e-05_run1_seed0/checkpoint_6.pt \
--alpha 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.85 0.9 0.95 \
--losses ce hinge entropy dcm flyp l1zero l2zero l1init l2init --template fmow_template