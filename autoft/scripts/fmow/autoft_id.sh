#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --account=iris # Run on IRIS nodes
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris7
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=64G # Request 16GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name="fmow-autoft-flyp-10inner-100ep-1000ex-id" # Name the job (for easier monitoring)
#SBATCH --output=fmow-autoft-flyp-10inner-100ep-1000ex-id.log  # Name of the output log file
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

source /iris/u/cchoi1/robust-optimizer/ropt/bin/activate

cd ../..

export PYTHONPATH="${PYTHONPATH}:/iris/u/cchoi1/robust-optimizer/autoft/"

python3 src/main.py --method autoft --model ViT-B/16 --data-location /iris/u/yoonho/data/wilds \
--id FMOWTrain --ood FMOWIDVal --eval-datasets FMOWIDVal,FMOWIDTest,FMOWOODTest \
--num_ood_hp_examples 1000 --ft_epochs 20 \
--autoft_epochs 100 --inner_steps 10 --lr 1e-5 --wd 0.1 \
--batch-size 128 --accumulation_steps 2 --warmup_length 500 \
--losses ce hinge entropy dcm l1zero l2zero l1init l2init flyp \
--ft_data /iris/u/cchoi1/Data/csv/fmow_v1.1/fmow.csv \
--csv-img-key filepath --csv-caption-key title --get_labeled_csv \
--load /iris/u/cchoi1/robust-optimizer/autoft/zeroshot/clip_vitb16_fmow2.pt \
--workers 2