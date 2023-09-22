#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --account=iris # Run on IRIS nodes
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris7
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=32GB # Request 16GB of memory
#SBATCH --gres=gpu:2 # Request one GPU
#SBATCH --job-name="imagenet-autoft-layer-100ex" # Name the job (for easier monitoring)
#SBATCH --output=imagenet-autoft-layer-100ex.log  # Name of the output log file
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

source /iris/u/cchoi1/robust-optimizer/ropt/bin/activate

cd ../..

export PYTHONPATH="${PYTHONPATH}:/iris/u/cchoi1/robust-optimizer/autoft/"

python src/main.py --method autoft --model ViT-B/16 --loss_type LayerwiseLoss --data-location /iris/u/yoonho/data \
--id ImageNet --ood ImageNetC --eval-datasets ImageNetA,ImageNetR,ImageNetSketch,ImageNetV2,ObjectNet,ImageNet,ImageNetC \
--ft_epochs 10 --inner_steps 10 --autoft_epochs 10000 --val_freq 10 \
--lr 1e-5 --wd 0.1 --batch-size 512 --warmup_length 1000 --accumulation_steps 1 \
--load /zeroshot/clip_vitb16_imagenet.pt \
--num_ood_hp_examples 100