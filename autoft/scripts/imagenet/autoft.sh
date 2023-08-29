#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --account=iris # Run on IRIS nodes
#SBATCH --exclude=iris7
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=32G # Request 16GB of memory
#SBATCH --gres=gpu:2 # Request one GPU
#SBATCH --job-name="imagenet-autoft" # Name the job (for easier monitoring)
#SBATCH --output=imagenet-autoft.log  # Name of the output log file
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

source /iris/u/cchoi1/robust-optimizer/ropt/bin/activate

cd ../..

export PYTHONPATH="${PYTHONPATH}:/iris/u/cchoi1/robust-optimizer/autoft/"

python src/main.py --method autoft --model ViT-B/16 --data-location /iris/u/yoonho/data \
--id ImageNet --ood ImageNetC --eval-datasets ImageNetA,ImageNetR,ImageNetSketch,ImageNetV2,ObjectNet,ImageNet,ImageNetC \
--ft_epochs 10 --inner_steps 10 --autoft_epochs 10 --val_freq 10 --lr 7.5e-6 --wd 0.1 --batch-size 128 --workers 2 \
--load ./zeroshot/clip_vitb16_imagenet.pt --save autoft- --exp_name ImageNet/autoft \
--num_ood_examples 10000 --num_ood_hp_examples 100 --results-db ./results/ImageNet/autoft --warmup_length 2000