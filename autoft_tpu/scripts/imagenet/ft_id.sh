#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --account=iris # Run on IRIS nodes
#SBATCH --exclude=iris7
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=16G # Request 16GB of memory
#SBATCH --gres=gpu:2 # Request one GPU
#SBATCH --job-name="imagenet-ft-id" # Name the job (for easier monitoring)
#SBATCH --output=imagenet-ft-id.log  # Name of the output log file
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

source /iris/u/cchoi1/robust-optimizer/ropt/bin/activate

cd ../..

export PYTHONPATH="${PYTHONPATH}:/iris/u/cchoi1/robust-optimizer/autoft/"

python3 src/main.py --method ft-id --model ViT-B/16 --data-location /home/carolinechoi/robust-ft \
--id ImageNet --ood ImageNetC --eval-datasets ImageNetA,ImageNetR,ImageNetSketch,ImageNetV2,ObjectNet,ImageNet,ImageNetC \
--ft_epochs 10 --lr 3e-5 --wd 0.1 --batch-size 16 --accumulation_steps 4 --warmup_length 1000 \
--load /home/carolinechoi/robust-ft/zeroshot/clip_vitb16_imagenet.pt \
--num_ood_hp_examples 15000