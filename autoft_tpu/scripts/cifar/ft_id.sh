#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --account=iris # Run on IRIS nodes
#SBATCH --exclude=iris7
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=16G # Request 16GB of memory
#SBATCH --gres=gpu:2
#SBATCH --job-name="cifar-ft-id-plot" # Name the job (for easier monitoring)
#SBATCH --output=cifar-ft-id-plot.log  # Name of the output log file
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

source /iris/u/cchoi1/robust-optimizer/ropt/bin/activate

cd ../..

export PYTHONPATH="${PYTHONPATH}:/iris/u/cchoi1/robust-optimizer/autoft/"

python3 src/main.py --method ft-id --plot --model ViT-L/14 \
--data-location /home/carolinechoi/robust-ft \
--id CIFAR10 --ood CIFAR10C --eval-datasets CIFAR101,CIFAR102,CIFAR10,CIFAR10C \
--lr 3e-5 --wd 0.1 --batch-size 16 --warmup_length 1000 \
--load /home/carolinechoi/robust-ft/zeroshot/clip_vitl14_openai_cifar10.pt \
--num_ood_hp_examples 190 --ft_epochs 10 --accumulation_steps 2