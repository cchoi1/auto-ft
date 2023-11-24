#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --account=iris # Run on IRIS nodes
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris7
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=64G # Request 16GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name="iwildcam-weight-ensemble-plot" # Name the job (for easier monitoring)
#SBATCH --output=iwildcam-weight-ensemble-plot.log  # Name of the output log file
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

source /iris/u/cchoi1/robust-optimizer/ropt/bin/activate

cd ../..

export PYTHONPATH="${PYTHONPATH}:/iris/u/cchoi1/robust-optimizer/autoft/"

#python3 src/wise_ft.py --method autoft --model ViT-B/16 --data-location /iris/u/cchoi1/Data \
#--id IWildCamTrain --ood IWildCamOODVal --eval-datasets IWildCamIDTest,IWildCamOODTest \
#--num_ood_hp_examples 1000 --autoft_epochs 200 --inner_steps 10 --ft_epochs 20 \
#--lr 1e-5 --wd 0.1 --batch-size 256 --warmup_length 500 --workers 2 \
#--load /iris/u/cchoi1/robust-optimizer/autoft/zeroshot/clip_vitb16_iwildcam2.pt,/iris/u/cchoi1/robust-optimizer/autoft/saved/IWildCamTrain/autoft/oodIWildCamOODVal___flyp/no1000_nouNone_afep500_is10_ftep20_bs128_wd0.1_lr1e-05_run1_seed0/checkpoint_14.pt \
#--alpha 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.85 0.9 0.95 1.0 \
#--losses ce dcm flyp --clip_gradient --template iwildcam_template

python3 src/wise_ft.py --method autoft --model ViT-B/16 --data-location /iris/u/cchoi1/Data \
--id IWildCamTrain --ood IWildCamOODVal --eval-datasets IWildCamIDTest,IWildCamOODTest \
--num_ood_hp_examples 1000 --autoft_epochs 200 --inner_steps 10 --ft_epochs 20 \
--lr 1e-5 --wd 0.1 --batch-size 256 --warmup_length 500 --workers 2 \
--load /iris/u/cchoi1/robust-optimizer/autoft/zeroshot/clip_vitl14_iwildcam336px.pt,/iris/u/asc8/workspace/robust-optimizer/autoft/saved/IWildCamTrain/autoft/oodIWildCamOODVal_cdf/no1000_afep500_is100_ftep20_bs64_wd0.0_lr1e-05_run1_seed0_ViT-L/14/checkpoint_16.pt \
--alpha 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.85 0.9 0.95 1.0 \
--losses ce dcm flyp --clip_gradient --template iwildcam_template