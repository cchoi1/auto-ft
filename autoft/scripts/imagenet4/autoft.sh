#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --account=iris # Run on IRIS nodes
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris7
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=64GB # Request 16GB of memory
#SBATCH --gres=gpu:2 # Request one GPU
#SBATCH --job-name="imagenet4-autoft-10is-1000evals-15000ex" # Name the job (for easier monitoring)
#SBATCH --output=imagenet4-autoft-10is-1000evals-15000ex.log  # Name of the output log file
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

source /iris/u/cchoi1/robust-optimizer/ropt/bin/activate

cd ../..

export PYTHONPATH="${PYTHONPATH}:/iris/u/cchoi1/robust-optimizer/autoft/"

python3 src/main.py --method autoft --model ViT-B/16 --data-location /iris/u/yoonho/data \
--id ImageNet4 --ood ImageNetC --eval-datasets ImageNetA,ImageNetR,ImageNetSketch,ImageNetV2,ObjectNet,ImageNet4,ImageNetC \
--ft_epochs 10 --inner_steps 10 --autoft_epochs 100 --val_freq 10 \
--lr 1e-5 --wd 0.1 --batch-size 512 --warmup_length 500 --accumulation_steps 1 \
--load ./zeroshot/clip_vitb16_imagenet.pt \
--num_ood_hp_examples 10000