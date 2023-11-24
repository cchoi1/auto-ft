#!/bin/bash
#SBATCH --account=iris # Run on IRIS nodes
#SBATCH --exclude=iris1,iris2,iris3,iris4
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=16G # Request 16GB of memory
#SBATCH --gres=gpu:1
#SBATCH --job-name="cifar-vitb16-lp" # Name the job (for easier monitoring)
#SBATCH --output=cifar-vitb16-lp.log  # Name of the output log file
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

source /iris/u/cchoi1/robust-optimizer/ropt/bin/activate

cd ../..

export PYTHONPATH="${PYTHONPATH}:/iris/u/cchoi1/robust-optimizer/autoft/"

python src/main.py --method ft-id --model ViT-B/16 --data-location /iris/u/cchoi1/Data \
--id CIFAR10 --ood CIFAR10C --eval-datasets CIFAR101,CIFAR102,CIFAR10 \
--lr 1e-3 --wd 0.2 --batch-size 128 --accumulation_steps 2 --warmup_length 500 \
--num_ood_hp_examples -1 --ft_epochs 20 --freeze-encoder \
--load /iris/u/cchoi1/robust-optimizer/autoft/zeroshot/clip_vitb16_cifar10.pt \
--losses ce --no_regenerate_head