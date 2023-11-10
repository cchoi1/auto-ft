#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --account=iris # Run on IRIS nodes
#SBATCH --exclude=iris1,iris2,iris3,iris4
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=64G # Request 16GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name="stanfordcars-autoft-500inner-300ep-1000ex-class-balanced-layer" # Name the job (for easier monitoring)
#SBATCH --output=stanfordcars-autoft-500inner-300ep-1000ex-class-balanced-layer.log  # Name of the output log file
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

source /iris/u/cchoi1/robust-optimizer/ropt/bin/activate

cd ../..

export PYTHONPATH="${PYTHONPATH}:/iris/u/cchoi1/robust-optimizer/autoft/"

python3 src/main.py --method autoft --model ViT-B/16 --data-location /iris/u/cchoi1/Data \
--id StanfordCarsTrain --ood StanfordCarsValHOpt --eval-datasets StanfordCarsValEarlyStopping,StanfordCarsTest \
--num_ood_hp_examples 1000 --use_class_balanced_ood --ft_epochs 100 \
--autoft_epochs 300 --inner_steps 500 --lr 1e-5 --wd 0.0 \
--batch-size 128 --accumulation_steps 2 --warmup_length 500 \
--losses ce dcm entropy flyp hinge l1init l1zero l2init l2zero \
--ft_data /iris/u/cchoi1/Data/stanford-cars/train.csv --template stanfordcars_template \
--csv-img-key filepath --csv-caption-key title --get_labeled_csv \
--load /iris/u/cchoi1/robust-optimizer/autoft/zeroshot/clip_vitb16_stanfordcars.pt \
--workers 2 --relative_to_flyp --regenerate_head --no_regenerate_head \
--layerwise_loss --layerwise_opt