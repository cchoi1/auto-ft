#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --account=iris # Run on IRIS nodes
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris7,iris-hp-z8
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=16G # Request 16GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name="iwildcam-corr-layer-300" # Name the job (for easier monitoring)
#SBATCH --output=iwildcam-corr-layer-300.log  # Name of the output log file
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=yoonho@stanford.edu     # Where to send mail

bash /iris/u/yoonho/slurm_print_info.sh
source /iris/u/yoonho/env/bin/activate

export PYTHONPATH="${PYTHONPATH}:/iris/u/yoonho/robust-optimizer/autoft/"

python3 src/main.py --method autoft --model ViT-B/16 --layerwise_loss --data-location /iris/u/cchoi1/Data \
--id IWildCamTrain --ood IWildCamOODVal --eval-datasets IWildCamIDVal,IWildCamIDTest,IWildCamOODTest \
--num_ood_hp_examples 512 --ft_epochs 20 \
--autoft_epochs 200 --inner_steps 10 --inner_loop_val_steps 3 10 30 100 300 --lr 1e-5 --wd 0.1 \
--batch-size 32 --warmup_length 500 --workers 2 \
--losses ce hinge entropy dcm l1zero l2zero l1init l2init \
--load /iris/u/cchoi1/robust-optimizer/autoft/zeroshot/clip_vitb16_iwildcam.pt